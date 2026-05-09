"""RSI mean-reversion strategy (long-only).

Buys when the RSI(period) drops below ``oversold``, exits when it rises
above ``overbought``. Uses Wilder's smoothing.
"""

from __future__ import annotations

import pandas as pd

from bybit_backtest.strategies.base import Order, OrderSide, Strategy, StrategyContext


class RSIMeanReversionStrategy(Strategy):
    name = "rsi_mr"

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        if period < 2:
            raise ValueError("period must be at least 2")
        if not 0 < oversold < overbought < 100:
            raise ValueError("expected 0 < oversold < overbought < 100")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self._prev_close: float | None = None
        self._avg_gain: float = 0.0
        self._avg_loss: float = 0.0
        self._samples: int = 0

    def reset(self) -> None:
        self._prev_close = None
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._samples = 0

    def describe(self) -> dict:
        return {
            "name": self.name,
            "period": self.period,
            "oversold": self.oversold,
            "overbought": self.overbought,
        }

    def _update_rsi(self, close: float) -> float | None:
        if self._prev_close is None:
            self._prev_close = close
            return None
        change = close - self._prev_close
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        self._prev_close = close
        self._samples += 1
        if self._samples <= self.period:
            # Build up the simple average of the first ``period`` bars.
            self._avg_gain += gain / self.period
            self._avg_loss += loss / self.period
            if self._samples < self.period:
                return None
        else:
            # Wilder's smoothing.
            self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
            self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period
        if self._avg_loss == 0:
            return 100.0
        rs = self._avg_gain / self._avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def on_bar(self, bar: pd.Series, ctx: StrategyContext) -> list[Order]:
        rsi = self._update_rsi(float(bar["close"]))
        if rsi is None:
            return []
        orders: list[Order] = []
        if rsi < self.oversold and ctx.position == 0 and ctx.cash > 0:
            orders.append(Order(side=OrderSide.BUY, cash_fraction=1.0))
        elif rsi > self.overbought and ctx.position > 0:
            orders.append(Order(side=OrderSide.SELL, position_fraction=1.0))
        return orders
