"""MACD crossover strategy (long-only).

Computes the classic MACD indicator (Moving Average Convergence Divergence)
with EMA(fast) - EMA(slow) and an EMA(signal) of the MACD line. Goes fully
long when the MACD line crosses above the signal line, and exits when it
crosses below.
"""

from __future__ import annotations

import pandas as pd

from bybit_backtest.strategies.base import Order, OrderSide, Strategy, StrategyContext


def _ema_alpha(period: int) -> float:
    return 2.0 / (period + 1.0)


class MACDStrategy(Strategy):
    """Classic MACD crossover (12/26/9 by default)."""

    name = "macd"

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> None:
        if fast <= 0 or slow <= 0 or signal <= 0:
            raise ValueError("fast, slow and signal periods must be positive")
        if fast >= slow:
            raise ValueError("fast period must be strictly smaller than slow period")
        self.fast = int(fast)
        self.slow = int(slow)
        self.signal = int(signal)
        self._fast_ema: float | None = None
        self._slow_ema: float | None = None
        self._signal_ema: float | None = None
        self._bars_seen: int = 0
        self._prev_diff: float | None = None

    def reset(self) -> None:
        self._fast_ema = None
        self._slow_ema = None
        self._signal_ema = None
        self._bars_seen = 0
        self._prev_diff = None

    def describe(self) -> dict:
        return {
            "name": self.name,
            "fast": self.fast,
            "slow": self.slow,
            "signal": self.signal,
        }

    def _update_ema(self, current: float | None, value: float, alpha: float) -> float:
        if current is None:
            return value
        return alpha * value + (1.0 - alpha) * current

    def on_bar(self, bar: pd.Series, ctx: StrategyContext) -> list[Order]:
        close = float(bar["close"])
        self._bars_seen += 1

        a_fast = _ema_alpha(self.fast)
        a_slow = _ema_alpha(self.slow)
        a_sig = _ema_alpha(self.signal)

        self._fast_ema = self._update_ema(self._fast_ema, close, a_fast)
        self._slow_ema = self._update_ema(self._slow_ema, close, a_slow)

        # Wait until the slow EMA has had time to stabilise.
        if self._bars_seen < self.slow:
            return []

        macd_line = self._fast_ema - self._slow_ema
        self._signal_ema = self._update_ema(self._signal_ema, macd_line, a_sig)

        if self._bars_seen < self.slow + self.signal:
            return []

        diff = macd_line - self._signal_ema
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
