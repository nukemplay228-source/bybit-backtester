"""Bollinger Bands mean-reversion strategy (long-only).

Buys when the close drops below the lower band (price ``num_std`` standard
deviations under the moving average) and exits when it climbs back above
the middle band (the moving average itself). Best suited for ranging /
sideways markets.
"""

from __future__ import annotations

import math
from collections import deque

import pandas as pd

from bybit_backtest.strategies.base import Order, OrderSide, Strategy, StrategyContext


class BollingerBandsStrategy(Strategy):
    """Mean-reversion on Bollinger Bands."""

    name = "bollinger"

    def __init__(self, period: int = 20, num_std: float = 2.0) -> None:
        if period < 2:
            raise ValueError("period must be at least 2")
        if num_std <= 0:
            raise ValueError("num_std must be positive")
        self.period = int(period)
        self.num_std = float(num_std)
        self._buf: deque[float] = deque(maxlen=self.period)

    def reset(self) -> None:
        self._buf.clear()

    def describe(self) -> dict:
        return {"name": self.name, "period": self.period, "num_std": self.num_std}

    def _stats(self) -> tuple[float, float]:
        n = len(self._buf)
        mean = sum(self._buf) / n
        var = sum((x - mean) ** 2 for x in self._buf) / n
        return mean, math.sqrt(var)

    def on_bar(self, bar: pd.Series, ctx: StrategyContext) -> list[Order]:
        close = float(bar["close"])
        self._buf.append(close)
        if len(self._buf) < self.period:
            return []
        mean, std = self._stats()
        lower = mean - self.num_std * std

        orders: list[Order] = []
        # Entry: price punctures the lower band.
        if close < lower and ctx.position == 0 and ctx.cash > 0:
            orders.append(Order(side=OrderSide.BUY, cash_fraction=1.0))
        # Exit: price rallies back to (or above) the mean — locks in the
        # mean-reversion edge instead of waiting for a full upper-band touch.
        elif close >= mean and ctx.position > 0:
            orders.append(Order(side=OrderSide.SELL, position_fraction=1.0))
        return orders
