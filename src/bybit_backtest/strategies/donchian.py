"""Donchian channel breakout strategy (long-only).

Inspired by the original Turtle Traders system. The strategy enters long
when the close breaks above the highest close of the previous
``entry_period`` bars, and exits when it drops below the lowest close of
the previous ``exit_period`` bars. ``exit_period`` is typically set
shorter than ``entry_period`` to lock in profit faster.
"""

from __future__ import annotations

from collections import deque

import pandas as pd

from bybit_backtest.strategies.base import Order, OrderSide, Strategy, StrategyContext


class DonchianBreakoutStrategy(Strategy):
    """Classic Turtle-style breakout."""

    name = "donchian"

    def __init__(self, entry_period: int = 55, exit_period: int = 20) -> None:
        if entry_period < 2 or exit_period < 2:
            raise ValueError("entry_period and exit_period must be at least 2")
        self.entry_period = int(entry_period)
        self.exit_period = int(exit_period)
        self._entry_buf: deque[float] = deque(maxlen=self.entry_period)
        self._exit_buf: deque[float] = deque(maxlen=self.exit_period)

    def reset(self) -> None:
        self._entry_buf.clear()
        self._exit_buf.clear()

    def describe(self) -> dict:
        return {
            "name": self.name,
            "entry_period": self.entry_period,
            "exit_period": self.exit_period,
        }

    def on_bar(self, bar: pd.Series, ctx: StrategyContext) -> list[Order]:
        close = float(bar["close"])

        # Compare with the prior window (exclude the current close).
        entry_high = max(self._entry_buf) if self._entry_buf else None
        exit_low = min(self._exit_buf) if self._exit_buf else None

        # Update buffers AFTER reading their current state so we always
        # compare against the prior N bars (no peeking at the present bar).
        self._entry_buf.append(close)
        self._exit_buf.append(close)

        orders: list[Order] = []
        if entry_high is None or exit_low is None:
            return orders
        if len(self._entry_buf) < self.entry_period:
            # Wait until we have a full lookback window.
            return orders

        if ctx.position == 0 and ctx.cash > 0 and close > entry_high:
            orders.append(Order(side=OrderSide.BUY, cash_fraction=1.0))
        elif ctx.position > 0 and close < exit_low:
            orders.append(Order(side=OrderSide.SELL, position_fraction=1.0))
        return orders
