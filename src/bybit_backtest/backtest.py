"""Vectorised-ish bar-by-bar backtest engine.

The engine iterates over OHLCV bars in chronological order. On each bar:

1. The bar is *closed*; the strategy looks at it and emits orders.
2. Orders are queued and executed at the *next* bar's open price with a
   configurable slippage adjustment and the portfolio's fee rate.
3. Equity is recorded using the close price of the current bar.

This avoids look-ahead bias: decisions only see information that would
have been available at close.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from bybit_backtest.metrics import PerformanceReport, compute_report
from bybit_backtest.portfolio import Portfolio
from bybit_backtest.strategies.base import Order, OrderSide, Strategy, StrategyContext

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest output."""

    equity: pd.Series
    trades: pd.DataFrame
    report: PerformanceReport
    strategy_params: dict


def _apply_slippage(price: float, side: OrderSide, slippage: float) -> float:
    if side == OrderSide.BUY:
        return price * (1.0 + slippage)
    return price * (1.0 - slippage)


def _execute_orders(
    portfolio: Portfolio,
    pending: list[Order],
    open_price: float,
    timestamp: pd.Timestamp,
    slippage: float,
) -> None:
    for order in pending:
        exec_price = _apply_slippage(open_price, order.side, slippage)
        if order.side == OrderSide.BUY:
            cash_to_spend = portfolio.cash * max(min(order.cash_fraction, 1.0), 0.0)
            if cash_to_spend > 0:
                portfolio.buy(timestamp.to_pydatetime(), exec_price, cash_to_spend)
        else:
            qty = portfolio.position * max(min(order.position_fraction, 1.0), 0.0)
            if qty > 0:
                portfolio.sell(timestamp.to_pydatetime(), exec_price, qty)


def run_backtest(
    bars: pd.DataFrame,
    strategy: Strategy,
    *,
    initial_cash: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage: float = 0.0005,
    interval: str = "60",
) -> BacktestResult:
    """Run ``strategy`` on ``bars`` and return the result.

    ``bars`` must have a UTC :class:`~pandas.DatetimeIndex` and the columns
    ``open, high, low, close, volume``. Anything else is ignored.
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars is missing required columns: {sorted(missing)}")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("bars must have a DatetimeIndex")
    if bars.index.tz is None:
        raise ValueError("bars index must be timezone-aware (UTC)")
    if len(bars) < 2:
        raise ValueError("need at least 2 bars to run a backtest")

    strategy.reset()
    portfolio = Portfolio(initial_cash=initial_cash, fee_rate=fee_rate)

    timestamps: list[pd.Timestamp] = list(bars.index)
    closes = bars["close"].to_numpy(dtype="float64")
    opens = bars["open"].to_numpy(dtype="float64")

    pending: list[Order] = []

    for i in range(len(bars)):
        ts = timestamps[i]
        # 1. Execute orders queued by the previous close at this bar's open.
        if pending:
            _execute_orders(
                portfolio,
                pending,
                open_price=float(opens[i]),
                timestamp=ts,
                slippage=slippage,
            )
            pending = []
        # 2. Record equity at this bar's close (mark-to-market).
        portfolio.record_equity(ts.to_pydatetime(), float(closes[i]))
        # 3. Ask the strategy for new orders for the *next* bar's open.
        if i < len(bars) - 1:
            ctx = StrategyContext(
                cash=portfolio.cash,
                position=portfolio.position,
                equity=portfolio.equity(float(closes[i])),
                avg_entry_price=portfolio.avg_entry_price,
            )
            new_orders = strategy.on_bar(bars.iloc[i], ctx)
            if new_orders:
                pending.extend(new_orders)

    equity_series = portfolio.equity_series()
    trades_df = portfolio.trades_dataframe()
    report = compute_report(equity_series, trades_df, interval=interval)
    return BacktestResult(
        equity=equity_series,
        trades=trades_df,
        report=report,
        strategy_params=strategy.describe(),
    )
