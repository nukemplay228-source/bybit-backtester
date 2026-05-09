"""Two-leg perpetual pairs-trading (statistical arbitrage) backtest.

The strategy bets on **mean reversion of a price spread** between two
correlated perpetuals (e.g. BTC and ETH). At every 8h bar:

* compute ``spread_t = log(close_a_t) - log(close_b_t)``;
* compute the rolling mean and std of the spread over ``lookback`` bars,
  then ``z_t = (spread_t - mean_t) / std_t``;
* if currently flat and ``z_t > +entry_z``: **short the spread** —
  short A, long B (asset A is "rich" relative to B);
* if currently flat and ``z_t < -entry_z``: **long the spread** —
  long A, short B (asset A is "cheap");
* if currently in position and ``|z_t| < exit_z``: close both legs.

Both legs are USDT-margined perpetuals with equal notional, so the
position is approximately **delta-neutral** — direct price exposure
on A is offset by an opposite price exposure on B. Day-to-day P&L
comes from the spread relaxing back toward the mean, **plus** funding
on the two perp legs (long pays positive funding, short receives), and
**minus** entry/exit fees and slippage.

Margin and liquidation are modelled the same way as ``funding_arb``:
both legs post initial margin; cross-margin treats the combined leg
P&L as collateral for the maintenance check; isolated-margin treats
each leg's collateral separately and can liquidate independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PairsArbConfig:
    """Knobs for the pairs-trading simulation."""

    lookback: int = 30
    entry_z: float = 2.0
    exit_z: float = 0.5
    notional_per_leg: float = 1000.0
    initial_margin_rate: float = 0.10
    maintenance_margin_rate: float = 0.05
    fee_rate: float = 0.0006
    slippage: float = 0.0005
    margin_mode: str = "cross"

    def __post_init__(self) -> None:
        if self.lookback < 5:
            raise ValueError("lookback must be at least 5 bars")
        if not 0 < self.exit_z < self.entry_z:
            raise ValueError("must satisfy 0 < exit_z < entry_z")
        if self.notional_per_leg <= 0:
            raise ValueError("notional_per_leg must be positive")
        if not 0 < self.maintenance_margin_rate < self.initial_margin_rate <= 1.0:
            raise ValueError(
                "must satisfy 0 < maintenance_margin_rate < initial_margin_rate <= 1.0"
            )
        for name in ("fee_rate", "slippage"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.margin_mode not in {"cross", "isolated"}:
            raise ValueError("margin_mode must be 'cross' or 'isolated'")


@dataclass
class PairsArbResult:
    """Outcome of a pairs-trading backtest."""

    equity: pd.Series
    z_score: pd.Series
    spread: pd.Series
    position: pd.Series
    funding_payments: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)
    liquidated: bool = False
    liquidation_time: pd.Timestamp | None = None


def _align_pairs(
    a_klines: pd.DataFrame,
    b_klines: pd.DataFrame,
    a_funding: pd.DataFrame,
    b_funding: pd.DataFrame,
) -> pd.DataFrame:
    """Align both perp price feeds and both funding feeds onto a common grid.

    Funding settles every 8h on Bybit/KuCoin; klines align to a slightly
    different 8h grid (00/08/16 vs 04/12/20 UTC). We resolve the offset by
    iterating the funding timestamps of the **first** asset and taking the
    most recent kline + funding data already published for both legs at
    each step.
    """
    for label, frame in (
        ("a_klines", a_klines),
        ("b_klines", b_klines),
        ("a_funding", a_funding),
        ("b_funding", b_funding),
    ):
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError(f"{label} must have a timezone-aware DatetimeIndex")

    def _flatten_kline(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        cols = ["open", "close"]
        renamed = frame.sort_index()[cols].rename(
            columns={"open": f"{prefix}_open", "close": f"{prefix}_close"}
        )
        return renamed.reset_index().rename(
            columns={renamed.index.name or "index": "ts"}
        )

    def _flatten_funding(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        renamed = frame.sort_index()[["funding_rate"]].rename(
            columns={"funding_rate": f"{prefix}_funding_rate"}
        )
        return renamed.reset_index().rename(
            columns={renamed.index.name or "index": "ts"}
        )

    a_kl = _flatten_kline(a_klines, "a")
    b_kl = _flatten_kline(b_klines, "b")
    a_fund = _flatten_funding(a_funding, "a")
    b_fund = _flatten_funding(b_funding, "b")

    base = a_fund.copy()
    for right in (b_fund, a_kl, b_kl):
        base = pd.merge_asof(base, right, on="ts", direction="backward")

    base = base.dropna(
        subset=[
            "a_open",
            "a_close",
            "b_open",
            "b_close",
            "a_funding_rate",
            "b_funding_rate",
        ]
    ).copy()
    base = base.set_index("ts").sort_index()
    base.index.name = "timestamp"
    return base


def _compute_z(spread: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = pd.Series(spread)
    mean = s.rolling(lookback).mean().to_numpy()
    std = s.rolling(lookback).std(ddof=0).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (spread - mean) / std
    return mean, std, z


def run_pairs_arb(
    a_klines: pd.DataFrame,
    b_klines: pd.DataFrame,
    a_funding: pd.DataFrame,
    b_funding: pd.DataFrame,
    config: PairsArbConfig | None = None,
) -> PairsArbResult:
    """Simulate the pairs-trading strategy on aligned 8h bars."""
    config = config or PairsArbConfig()

    df = _align_pairs(a_klines, b_klines, a_funding, b_funding)
    if len(df) <= config.lookback:
        raise ValueError(
            f"need more than lookback={config.lookback} aligned bars, got {len(df)}"
        )

    n = len(df)
    a_close = df["a_close"].to_numpy()
    b_close = df["b_close"].to_numpy()
    a_fund_rate = df["a_funding_rate"].to_numpy()
    b_fund_rate = df["b_funding_rate"].to_numpy()
    timestamps = df.index

    spread = np.log(a_close) - np.log(b_close)
    _, _, z = _compute_z(spread, config.lookback)

    equity_series = np.zeros(n, dtype="float64")
    position_series = np.zeros(n, dtype="int8")
    funding_series = np.zeros(n, dtype="float64")
    z_series = z.copy()
    spread_series = spread.copy()

    trades: list[dict[str, object]] = []

    state = 0  # -1 short spread (short A, long B), +1 long spread, 0 flat
    a_qty = 0.0
    b_qty = 0.0
    a_entry = 0.0
    b_entry = 0.0
    a_dir = 0  # +1 long, -1 short
    b_dir = 0

    realised_pnl = 0.0
    margin_posted = 0.0
    cash_outlay = 0.0
    liquidated = False
    liquidation_time: pd.Timestamp | None = None

    def _open_position(i: int, direction: int) -> None:
        """direction = +1 (long spread: long A, short B) or -1 (short spread)."""
        nonlocal state, a_qty, b_qty, a_entry, b_entry, a_dir, b_dir
        nonlocal margin_posted, cash_outlay, realised_pnl

        a_dir = direction
        b_dir = -direction
        a_slip_sign = config.slippage * a_dir
        b_slip_sign = config.slippage * b_dir
        a_entry = a_close[i] * (1 + a_slip_sign)
        b_entry = b_close[i] * (1 + b_slip_sign)
        a_qty = config.notional_per_leg / a_entry
        b_qty = config.notional_per_leg / b_entry
        a_fee = a_qty * a_entry * config.fee_rate
        b_fee = b_qty * b_entry * config.fee_rate
        realised_pnl -= a_fee + b_fee
        leg_margin = config.notional_per_leg * config.initial_margin_rate
        margin_posted = 2 * leg_margin
        cash_outlay = margin_posted
        state = direction

        for prefix, qty, price, side, fee in (
            ("a", a_qty, a_entry, "BUY" if a_dir > 0 else "SELL", a_fee),
            ("b", b_qty, b_entry, "BUY" if b_dir > 0 else "SELL", b_fee),
        ):
            trades.append(
                {
                    "timestamp": timestamps[i],
                    "leg": prefix,
                    "side": side,
                    "price": price,
                    "qty": qty,
                    "fee": fee,
                    "event": "open",
                }
            )

    def _close_position(i: int, reason: str) -> None:
        nonlocal state, a_qty, b_qty, a_entry, b_entry, a_dir, b_dir
        nonlocal margin_posted, cash_outlay, realised_pnl

        a_slip_sign = -config.slippage * a_dir
        b_slip_sign = -config.slippage * b_dir
        a_exit = a_close[i] * (1 + a_slip_sign)
        b_exit = b_close[i] * (1 + b_slip_sign)
        a_pnl = a_dir * (a_exit - a_entry) * a_qty
        b_pnl = b_dir * (b_exit - b_entry) * b_qty
        a_fee = a_qty * a_exit * config.fee_rate
        b_fee = b_qty * b_exit * config.fee_rate
        realised_pnl += a_pnl + b_pnl - a_fee - b_fee

        for prefix, qty, price, side, fee in (
            ("a", a_qty, a_exit, "SELL" if a_dir > 0 else "BUY", a_fee),
            ("b", b_qty, b_exit, "SELL" if b_dir > 0 else "BUY", b_fee),
        ):
            trades.append(
                {
                    "timestamp": timestamps[i],
                    "leg": prefix,
                    "side": side,
                    "price": price,
                    "qty": qty,
                    "fee": fee,
                    "event": reason,
                }
            )

        state = 0
        a_qty = b_qty = 0.0
        a_dir = b_dir = 0
        margin_posted = 0.0
        cash_outlay = 0.0

    for i in range(n):
        if state != 0:
            f_a = -a_dir * a_fund_rate[i] * a_qty * a_close[i]
            f_b = -b_dir * b_fund_rate[i] * b_qty * b_close[i]
            realised_pnl += f_a + f_b
            funding_series[i] = f_a + f_b

        if state != 0:
            unreal_a = a_dir * (a_close[i] - a_entry) * a_qty
            unreal_b = b_dir * (b_close[i] - b_entry) * b_qty
            collateral = (
                margin_posted + realised_pnl + unreal_a + unreal_b
                if config.margin_mode == "cross"
                else margin_posted + min(unreal_a, 0.0) + min(unreal_b, 0.0)
            )
            notional_now = a_qty * a_close[i] + b_qty * b_close[i]
            maintenance_required = notional_now * config.maintenance_margin_rate
            if collateral <= maintenance_required and not liquidated:
                liquidated = True
                liquidation_time = timestamps[i]
                _close_position(i, "liquidation")

        if not liquidated and not np.isnan(z[i]):
            if state == 0:
                if z[i] > config.entry_z:
                    _open_position(i, direction=-1)
                elif z[i] < -config.entry_z:
                    _open_position(i, direction=+1)
            else:
                if abs(z[i]) <= config.exit_z:
                    _close_position(i, reason="signal_exit")

        if state != 0:
            unreal_a = a_dir * (a_close[i] - a_entry) * a_qty
            unreal_b = b_dir * (b_close[i] - b_entry) * b_qty
            equity_series[i] = realised_pnl + unreal_a + unreal_b
        else:
            equity_series[i] = realised_pnl
        position_series[i] = state

    if state != 0:
        _close_position(n - 1, reason="final_close")
        equity_series[n - 1] = realised_pnl

    duration_days = max(
        (timestamps[-1] - timestamps[0]).total_seconds() / 86400.0, 1.0
    )
    capital_at_risk = max(2.0 * config.notional_per_leg * config.initial_margin_rate, 1.0)
    total_return = realised_pnl / capital_at_risk
    annualised = (
        (1.0 + total_return) ** (365.25 / duration_days) - 1.0
        if total_return > -0.999
        else -1.0
    )
    rolling_max = np.maximum.accumulate(equity_series)
    drawdown = equity_series - rolling_max
    max_drawdown = float(drawdown.min()) / capital_at_risk if capital_at_risk > 0 else 0.0

    num_trade_pairs = sum(1 for t in trades if t["event"] != "open") // 2

    metrics = {
        "total_return": float(total_return),
        "annualised_return": float(annualised),
        "total_funding_received": float(funding_series.sum()),
        "duration_days": float(duration_days),
        "max_drawdown": max_drawdown,
        "num_round_trips": int(num_trade_pairs),
        "settlements": int(n),
        "capital_at_risk": float(capital_at_risk),
    }

    return PairsArbResult(
        equity=pd.Series(equity_series, index=timestamps, name="equity"),
        z_score=pd.Series(z_series, index=timestamps, name="z_score"),
        spread=pd.Series(spread_series, index=timestamps, name="spread"),
        position=pd.Series(position_series, index=timestamps, name="position"),
        funding_payments=pd.Series(funding_series, index=timestamps, name="funding"),
        trades=pd.DataFrame(trades),
        metrics=metrics,
        liquidated=liquidated,
        liquidation_time=liquidation_time,
    )
