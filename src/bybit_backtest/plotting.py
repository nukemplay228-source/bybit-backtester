"""Plotting helpers for backtest results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(
    equity: pd.Series,
    output_path: str | Path,
    *,
    title: str = "Equity curve",
    benchmark: pd.Series | None = None,
) -> Path:
    """Render equity + drawdown subplot to ``output_path`` (PNG)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_eq.plot(equity.index, equity.values, label="Strategy", color="#1f77b4", linewidth=1.5)
    if benchmark is not None and not benchmark.empty:
        ax_eq.plot(
            benchmark.index,
            benchmark.values,
            label="Buy & Hold",
            color="#888888",
            linewidth=1.0,
            linestyle="--",
        )
    ax_eq.set_title(title)
    ax_eq.set_ylabel("Equity")
    ax_eq.grid(True, alpha=0.3)
    ax_eq.legend(loc="best")

    running_peak = equity.cummax()
    drawdown = (equity - running_peak) / running_peak * 100
    ax_dd.fill_between(drawdown.index, drawdown.values, 0, color="#d62728", alpha=0.4)
    ax_dd.plot(drawdown.index, drawdown.values, color="#d62728", linewidth=0.8)
    ax_dd.set_ylabel("Drawdown %")
    ax_dd.set_xlabel("Date (UTC)")
    ax_dd.grid(True, alpha=0.3)

    locator = mdates.AutoDateLocator()
    ax_dd.xaxis.set_major_locator(locator)
    ax_dd.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def buy_and_hold_equity(bars: pd.DataFrame, initial_cash: float, fee_rate: float = 0.001) -> pd.Series:
    """Equity curve for a passive buy-and-hold benchmark."""
    if bars.empty:
        return pd.Series(dtype="float64")
    first_open = float(bars["open"].iloc[0])
    if first_open <= 0:
        return pd.Series(dtype="float64")
    quantity = (initial_cash * (1.0 - fee_rate)) / first_open
    equity = bars["close"] * quantity
    return equity.rename("benchmark")
