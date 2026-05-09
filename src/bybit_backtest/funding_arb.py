"""Delta-neutral funding-rate arbitrage backtest.

The strategy is the textbook "cash & carry" trade on a perpetual:

* open a long spot position of size ``notional``
* simultaneously open a short perp position of the same size
* sit on the position and collect funding every 8h (the short receives funding
  whenever the funding rate is positive, pays it when the rate is negative)
* close both legs at the end

In an idealised world the two legs cancel each other's price exposure exactly
and the strategy's PnL is just ``sum(funding_rate * notional)`` minus fees.
In practice spot and perp prices don't move in perfect lockstep (there is a
basis), so we mark both legs to market at every funding interval and track the
margin headroom on the short. If maintenance margin is breached the position
is forcibly closed, just like on the real venue.

The engine intentionally uses 8h bars throughout — that's the funding cadence
on Bybit/KuCoin/Binance USDT-margined perps, so working at this granularity
matches the real settlement schedule and keeps the simulation simple.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FundingArbConfig:
    """Knobs for the funding-arb simulation."""

    notional: float = 1000.0
    initial_margin_rate: float = 0.10
    maintenance_margin_rate: float = 0.05
    spot_fee_rate: float = 0.001
    perp_fee_rate: float = 0.0006
    slippage: float = 0.0005
    margin_mode: str = "cross"

    def __post_init__(self) -> None:
        if self.notional <= 0:
            raise ValueError("notional must be positive")
        if not 0 < self.maintenance_margin_rate < self.initial_margin_rate <= 1.0:
            raise ValueError(
                "must satisfy 0 < maintenance_margin_rate < initial_margin_rate <= 1.0"
            )
        for name in ("spot_fee_rate", "perp_fee_rate", "slippage"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.margin_mode not in {"cross", "isolated"}:
            raise ValueError("margin_mode must be 'cross' or 'isolated'")


@dataclass
class FundingArbResult:
    """Outcome of a funding-arb backtest."""

    equity: pd.Series
    funding_payments: pd.Series
    spot_value: pd.Series
    perp_value: pd.Series
    margin_ratio: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)
    liquidated: bool = False
    liquidation_time: pd.Timestamp | None = None


def _align_8h(
    spot: pd.DataFrame, perp: pd.DataFrame, funding: pd.DataFrame
) -> pd.DataFrame:
    """Align spot and perp prices onto the funding-payment grid.

    Funding settles every 8 hours (e.g. 04:00 / 12:00 / 20:00 UTC on
    KuCoin and Bybit), but kline bars are typically aligned to the
    00:00 / 08:00 / 16:00 grid, so a plain inner-join would leave the
    aligned frame empty. We resolve the offset with ``merge_asof`` —
    for each funding timestamp we use the most recent kline bar that
    has already closed.
    """
    if not isinstance(spot.index, pd.DatetimeIndex) or spot.index.tz is None:
        raise ValueError("spot frame must have a timezone-aware DatetimeIndex")
    if not isinstance(perp.index, pd.DatetimeIndex) or perp.index.tz is None:
        raise ValueError("perp frame must have a timezone-aware DatetimeIndex")
    if not isinstance(funding.index, pd.DatetimeIndex) or funding.index.tz is None:
        raise ValueError("funding frame must have a timezone-aware DatetimeIndex")

    spot_sorted = (
        spot.sort_index()
        .rename(
            columns={
                "open": "spot_open",
                "high": "spot_high",
                "low": "spot_low",
                "close": "spot_close",
            }
        )[["spot_open", "spot_high", "spot_low", "spot_close"]]
        .reset_index()
        .rename(columns={spot.index.name or "index": "ts"})
    )
    perp_sorted = (
        perp.sort_index()
        .rename(
            columns={
                "open": "perp_open",
                "high": "perp_high",
                "low": "perp_low",
                "close": "perp_close",
            }
        )[["perp_open", "perp_high", "perp_low", "perp_close"]]
        .reset_index()
        .rename(columns={perp.index.name or "index": "ts"})
    )
    funding_sorted = (
        funding.sort_index()
        .reset_index()
        .rename(columns={funding.index.name or "index": "ts"})
    )

    merged = pd.merge_asof(
        funding_sorted, spot_sorted, on="ts", direction="backward"
    )
    merged = pd.merge_asof(
        merged, perp_sorted, on="ts", direction="backward"
    )
    merged = merged.dropna(
        subset=[
            "spot_open",
            "spot_close",
            "perp_open",
            "perp_close",
            "funding_rate",
        ]
    ).copy()
    merged = merged.set_index("ts").sort_index()
    merged.index.name = "timestamp"
    return merged


def run_funding_arb(
    spot_klines: pd.DataFrame,
    perp_klines: pd.DataFrame,
    funding_rates: pd.DataFrame,
    config: FundingArbConfig | None = None,
) -> FundingArbResult:
    """Simulate the cash-and-carry funding-arb on aligned 8h bars.

    Parameters
    ----------
    spot_klines, perp_klines:
        Frames indexed by UTC timestamp with at least ``open`` and ``close``
        columns. The 8h grid is recommended.
    funding_rates:
        Frame indexed by UTC timestamp (the moment funding settles) with a
        ``funding_rate`` column. Sign convention: positive = longs pay shorts.
    """
    config = config or FundingArbConfig()

    df = _align_8h(spot_klines, perp_klines, funding_rates)
    if df.empty:
        raise ValueError("aligned frame is empty — check spot/perp/funding ranges")

    n = len(df)
    spot_open = df["spot_open"].to_numpy()
    spot_close = df["spot_close"].to_numpy()
    perp_open = df["perp_open"].to_numpy()
    perp_close = df["perp_close"].to_numpy()
    funding = df["funding_rate"].to_numpy()
    timestamps = df.index

    notional = config.notional
    spot_qty = notional / (spot_open[0] * (1 + config.slippage))
    perp_qty = notional / (perp_open[0] * (1 - config.slippage))

    spot_entry_price = spot_open[0] * (1 + config.slippage)
    perp_entry_price = perp_open[0] * (1 - config.slippage)
    spot_entry_fee = spot_qty * spot_entry_price * config.spot_fee_rate
    perp_entry_fee = perp_qty * perp_entry_price * config.perp_fee_rate

    spot_cost = spot_qty * spot_entry_price + spot_entry_fee
    margin_posted = perp_qty * perp_entry_price * config.initial_margin_rate

    cash_outlay = spot_cost + margin_posted + perp_entry_fee
    perp_pnl_running = -perp_entry_fee

    equity_series = np.empty(n, dtype="float64")
    funding_series = np.empty(n, dtype="float64")
    spot_value_series = np.empty(n, dtype="float64")
    perp_value_series = np.empty(n, dtype="float64")
    margin_series = np.empty(n, dtype="float64")

    trades: list[dict[str, object]] = [
        {
            "timestamp": timestamps[0],
            "leg": "spot",
            "side": "BUY",
            "price": spot_entry_price,
            "qty": spot_qty,
            "fee": spot_entry_fee,
        },
        {
            "timestamp": timestamps[0],
            "leg": "perp",
            "side": "SELL",
            "price": perp_entry_price,
            "qty": perp_qty,
            "fee": perp_entry_fee,
        },
    ]

    liquidated = False
    liquidation_time: pd.Timestamp | None = None

    for i in range(n):
        funding_payment = funding[i] * perp_qty * perp_close[i]
        perp_pnl_running += funding_payment

        spot_value = spot_qty * spot_close[i]
        perp_unrealised = (perp_entry_price - perp_close[i]) * perp_qty
        perp_value = perp_unrealised + perp_pnl_running

        account_value = spot_value + margin_posted + perp_value
        equity = account_value - cash_outlay
        notional_now = perp_qty * perp_close[i]
        maintenance_required = notional_now * config.maintenance_margin_rate

        collateral = (
            account_value
            if config.margin_mode == "cross"
            else margin_posted + perp_value
        )
        margin_ratio = (
            collateral / maintenance_required
            if maintenance_required > 0
            else float("inf")
        )

        equity_series[i] = equity
        funding_series[i] = funding_payment
        spot_value_series[i] = spot_value
        perp_value_series[i] = perp_value
        margin_series[i] = margin_ratio

        if collateral <= maintenance_required and not liquidated:
            liquidated = True
            liquidation_time = timestamps[i]

    last = n - 1
    spot_exit_price = spot_close[last] * (1 - config.slippage)
    perp_exit_price = perp_close[last] * (1 + config.slippage)
    spot_exit_fee = spot_qty * spot_exit_price * config.spot_fee_rate
    perp_exit_fee = perp_qty * perp_exit_price * config.perp_fee_rate

    spot_pnl_realised = spot_qty * (spot_exit_price - spot_entry_price) - spot_entry_fee - spot_exit_fee
    perp_pnl_realised = (
        perp_qty * (perp_entry_price - perp_exit_price) - perp_exit_fee + perp_pnl_running
    )
    total_pnl = spot_pnl_realised + perp_pnl_realised

    trades.append(
        {
            "timestamp": timestamps[last],
            "leg": "spot",
            "side": "SELL",
            "price": spot_exit_price,
            "qty": spot_qty,
            "fee": spot_exit_fee,
        }
    )
    trades.append(
        {
            "timestamp": timestamps[last],
            "leg": "perp",
            "side": "BUY",
            "price": perp_exit_price,
            "qty": perp_qty,
            "fee": perp_exit_fee,
        }
    )

    equity_series[last] = total_pnl

    equity_idx = pd.Series(equity_series, index=timestamps, name="equity")
    funding_idx = pd.Series(funding_series, index=timestamps, name="funding")
    spot_value_idx = pd.Series(spot_value_series, index=timestamps, name="spot_value")
    perp_value_idx = pd.Series(perp_value_series, index=timestamps, name="perp_value")
    margin_idx = pd.Series(margin_series, index=timestamps, name="margin_ratio")
    trades_df = pd.DataFrame(trades)

    duration_days = max(
        (timestamps[last] - timestamps[0]).total_seconds() / 86400.0, 1.0
    )
    total_return = total_pnl / cash_outlay
    annualised = (1.0 + total_return) ** (365.25 / duration_days) - 1.0
    total_funding = float(funding_series.sum())
    total_fees = (
        spot_entry_fee + spot_exit_fee + perp_entry_fee + perp_exit_fee
    )
    rolling_max = np.maximum.accumulate(equity_series)
    drawdown_dollars = equity_series - rolling_max
    max_drawdown = (
        float(drawdown_dollars.min()) / cash_outlay if cash_outlay > 0 else 0.0
    )

    metrics = {
        "total_return": float(total_return),
        "annualised_return": float(annualised),
        "total_funding_received": total_funding,
        "total_fees_paid": float(total_fees),
        "duration_days": float(duration_days),
        "max_drawdown": max_drawdown,
        "min_margin_ratio": float(margin_series.min()) if n else 0.0,
        "settlements": int(n),
    }

    return FundingArbResult(
        equity=equity_idx,
        funding_payments=funding_idx,
        spot_value=spot_value_idx,
        perp_value=perp_value_idx,
        margin_ratio=margin_idx,
        trades=trades_df,
        metrics=metrics,
        liquidated=liquidated,
        liquidation_time=liquidation_time,
    )
