"""Dollar-cost averaging strategy (long-only, buy-and-hold variant).

Buys a fixed dollar amount every ``interval_bars`` bars. Never sells —
this models the "set-and-forget" approach commonly recommended to retail
investors. Useful as a benchmark to compare more active strategies against.
"""

from __future__ import annotations

import pandas as pd

from bybit_backtest.strategies.base import Order, OrderSide, Strategy, StrategyContext


class DCAStrategy(Strategy):
    name = "dca"

    def __init__(self, interval_bars: int = 24, cash_per_buy: float = 10.0) -> None:
        if interval_bars <= 0:
            raise ValueError("interval_bars must be positive")
        if cash_per_buy <= 0:
            raise ValueError("cash_per_buy must be positive")
        self.interval_bars = int(interval_bars)
        self.cash_per_buy = float(cash_per_buy)
        self._counter = 0

    def reset(self) -> None:
        self._counter = 0

    def describe(self) -> dict:
        return {
            "name": self.name,
            "interval_bars": self.interval_bars,
            "cash_per_buy": self.cash_per_buy,
        }

    def on_bar(self, bar: pd.Series, ctx: StrategyContext) -> list[Order]:
        self._counter += 1
        if self._counter < self.interval_bars:
            return []
        self._counter = 0
        if ctx.cash <= 0:
            return []
        cash_fraction = min(self.cash_per_buy / ctx.cash, 1.0)
        return [Order(side=OrderSide.BUY, cash_fraction=cash_fraction)]
