"""Performance metrics for an equity curve and trade log."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Annualisation factor look-up keyed by Bybit interval string.
_BARS_PER_YEAR: dict[str, float] = {
    "1": 365 * 24 * 60,
    "3": 365 * 24 * 20,
    "5": 365 * 24 * 12,
    "15": 365 * 24 * 4,
    "30": 365 * 24 * 2,
    "60": 365 * 24,
    "120": 365 * 12,
    "240": 365 * 6,
    "360": 365 * 4,
    "720": 365 * 2,
    "D": 365,
    "W": 52,
    "M": 12,
}


@dataclass
class PerformanceReport:
    """Summary of strategy performance."""

    initial_equity: float
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    volatility_pct: float
    num_trades: int
    win_rate_pct: float
    profit_factor: float
    avg_trade_pct: float

    def to_dict(self) -> dict:
        return {
            "initial_equity": self.initial_equity,
            "final_equity": self.final_equity,
            "total_return_pct": self.total_return_pct,
            "cagr_pct": self.cagr_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "volatility_pct": self.volatility_pct,
            "num_trades": self.num_trades,
            "win_rate_pct": self.win_rate_pct,
            "profit_factor": self.profit_factor,
            "avg_trade_pct": self.avg_trade_pct,
        }

    def to_text(self) -> str:
        rows = [
            ("Initial equity", f"{self.initial_equity:,.2f}"),
            ("Final equity", f"{self.final_equity:,.2f}"),
            ("Total return", f"{self.total_return_pct:+.2f}%"),
            ("CAGR", f"{self.cagr_pct:+.2f}%"),
            ("Max drawdown", f"-{self.max_drawdown_pct:.2f}%"),
            ("Volatility (ann.)", f"{self.volatility_pct:.2f}%"),
            ("Sharpe", f"{self.sharpe:.2f}"),
            ("Sortino", f"{self.sortino:.2f}"),
            ("Trades", f"{self.num_trades}"),
            ("Win rate", f"{self.win_rate_pct:.2f}%"),
            ("Profit factor", f"{self.profit_factor:.2f}"),
            ("Avg trade", f"{self.avg_trade_pct:+.2f}%"),
        ]
        width = max(len(name) for name, _ in rows)
        return "\n".join(f"{name.ljust(width)}  {value}" for name, value in rows)


def bars_per_year(interval: str) -> float:
    if interval not in _BARS_PER_YEAR:
        raise ValueError(f"unknown interval {interval!r}")
    return _BARS_PER_YEAR[interval]


def max_drawdown(equity: pd.Series) -> float:
    """Return the maximum drawdown as a positive fraction (0.10 == 10%)."""
    if equity.empty:
        return 0.0
    running_peak = equity.cummax()
    dd = (equity - running_peak) / running_peak
    return float(-dd.min()) if len(dd) else 0.0


def _round_trip_returns(trades: pd.DataFrame) -> list[float]:
    """Pair BUY trades with SELL trades FIFO and compute pct returns.

    For grid-style strategies that interleave many buys and sells this
    treats each SELL as closing the oldest still-open BUY of the same
    quantity (allowing splits). Net of fees.
    """
    if trades.empty:
        return []
    open_buys: list[dict] = []  # each item: {price, qty_left, fee_per_unit}
    returns: list[float] = []
    for trade in trades.to_dict("records"):
        side = trade["side"]
        price = float(trade["price"])
        qty = float(trade["quantity"])
        fee = float(trade["fee"])
        if qty <= 0:
            continue
        if side == "BUY":
            open_buys.append(
                {"price": price, "qty_left": qty, "fee_per_unit": fee / qty}
            )
            continue
        # SELL: match against open buys FIFO.
        sell_remaining = qty
        sell_fee_per_unit = fee / qty
        while sell_remaining > 1e-12 and open_buys:
            head = open_buys[0]
            take = min(head["qty_left"], sell_remaining)
            buy_cost = (head["price"] + head["fee_per_unit"]) * take
            sell_proceeds = (price - sell_fee_per_unit) * take
            if buy_cost > 0:
                returns.append((sell_proceeds - buy_cost) / buy_cost)
            head["qty_left"] -= take
            sell_remaining -= take
            if head["qty_left"] <= 1e-12:
                open_buys.pop(0)
    return returns


def compute_report(
    equity: pd.Series,
    trades: pd.DataFrame,
    *,
    interval: str,
    risk_free_rate: float = 0.0,
) -> PerformanceReport:
    """Compute the full performance report from equity curve and trade log."""
    if equity.empty:
        return PerformanceReport(
            initial_equity=0.0,
            final_equity=0.0,
            total_return_pct=0.0,
            cagr_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe=0.0,
            sortino=0.0,
            volatility_pct=0.0,
            num_trades=0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            avg_trade_pct=0.0,
        )

    initial = float(equity.iloc[0])
    final = float(equity.iloc[-1])
    total_return = final / initial - 1.0 if initial > 0 else 0.0

    duration_seconds = (equity.index[-1] - equity.index[0]).total_seconds()
    years = duration_seconds / (365.25 * 24 * 3600)
    # CAGR is meaningless on a sub-day window and the math overflows quickly
    # (e.g. (1.21)**(1/2hours) is astronomical), so cap the annualisation.
    if initial > 0 and final > 0 and years >= 1 / 365:
        try:
            cagr = (final / initial) ** (1 / years) - 1.0
        except OverflowError:
            cagr = 0.0
    else:
        cagr = 0.0

    returns = equity.pct_change().dropna()
    bpy = bars_per_year(interval)
    if len(returns) > 1:
        mean_per_bar = float(returns.mean())
        std_per_bar = float(returns.std(ddof=1))
        downside = returns.clip(upper=0)
        downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
        rf_per_bar = risk_free_rate / bpy
        excess_per_bar = mean_per_bar - rf_per_bar
        sharpe = (
            (excess_per_bar / std_per_bar) * np.sqrt(bpy)
            if std_per_bar > 0
            else 0.0
        )
        sortino = (
            (excess_per_bar / downside_std) * np.sqrt(bpy)
            if downside_std > 0
            else 0.0
        )
        vol = std_per_bar * np.sqrt(bpy)
    else:
        sharpe = 0.0
        sortino = 0.0
        vol = 0.0

    dd = max_drawdown(equity)

    rt_returns = _round_trip_returns(trades)
    num_trades = len(rt_returns)
    if num_trades > 0:
        wins = [r for r in rt_returns if r > 0]
        losses = [r for r in rt_returns if r <= 0]
        win_rate = len(wins) / num_trades
        gross_win = sum(wins)
        gross_loss = -sum(losses)
        if gross_loss > 0:
            profit_factor = gross_win / gross_loss
        elif gross_win > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0
        avg_trade = sum(rt_returns) / num_trades
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade = 0.0

    return PerformanceReport(
        initial_equity=initial,
        final_equity=final,
        total_return_pct=total_return * 100,
        cagr_pct=cagr * 100,
        max_drawdown_pct=dd * 100,
        sharpe=sharpe,
        sortino=sortino,
        volatility_pct=vol * 100,
        num_trades=num_trades,
        win_rate_pct=win_rate * 100,
        profit_factor=profit_factor,
        avg_trade_pct=avg_trade * 100,
    )
