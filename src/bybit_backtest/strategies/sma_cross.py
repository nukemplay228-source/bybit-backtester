"""Simple moving average crossover strategy (long-only).

Goes fully long when the fast SMA crosses above the slow SMA, and exits the
entire position when the fast SMA crosses below.
"""

from __future__ import annotations

from collections import deque

import pandas as pd

from bybit_backtest.strategies.base import Order, OrderSide, Strategy, StrategyContext


class SMACrossStrategy(Strategy):
    """Classic golden/death cross on closes."""

    name = "sma_cross"

    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        if fast <= 0 or slow <= 0:
            raise ValueError("fast and slow windows must be positive")
        if fast >= slow:
            raise ValueError("fast window must be strictly smaller than slow window")
        self.fast = fast
        self.slow = slow
        self._fast_buf: deque[float] = deque(maxlen=fast)
        self._slow_buf: deque[float] = deque(maxlen=slow)
        self._prev_diff: float | None = None

    def reset(self) -> None:
        self._fast_buf.clear()
        self._slow_buf.clear()
        self._prev_diff = None

    def describe(self) -> dict:
        return {"name": self.name, "fast": self.fast, "slow": self.slow}

    def on_bar(self, bar: pd.Series, ctx: StrategyContext) -> list[Order]:
        close = float(bar["close"])
        self._fast_buf.append(close)
        self._slow_buf.append(close)
        if len(self._slow_buf) < self.slow:
            return []
        fast_avg = sum(self._fast_buf) / len(self._fast_buf)
        slow_avg = sum(self._slow_buf) / len(self._slow_buf)
        diff = fast_avg - slow_avg

        orders: list[Order] = []
        if self._prev_diff is not None:
            crossed_up = self._prev_diff <= 0 < diff
            crossed_down = self._prev_diff >= 0 > diff
            if crossed_up and ctx.position == 0 and ctx.cash > 0:
                orders.append(Order(side=OrderSide.BUY, cash_fraction=1.0))
            elif crossed_down and ctx.position > 0:
                orders.append(Order(side=OrderSide.SELL, position_fraction=1.0))
        self._prev_diff = diff
        return orders
