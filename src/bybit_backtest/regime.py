"""Regime-filter meta-strategy.

A regime filter wraps an arbitrary :class:`Strategy` and gates its long
entries by a slow moving-average trend filter. The intuition is simple:
many trend-following and even mean-reversion strategies bleed money in a
sustained bear market because every bounce they buy keeps fading. Ignoring
their long signals during such regimes — and forcing an exit if the regime
flips while a position is open — usually trades a chunk of the upside in
strong bull years for a much smaller drawdown in bear years.

The default filter is ``close > SMA(window)`` (e.g. ``window=200`` on daily
bars: the textbook 200-day filter from Brock/Lakonishok/LeBaron 1992 and
the de-facto industry baseline). Other regime definitions can be plugged
in by subclassing or by passing a custom ``regime_fn``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from bybit_backtest.strategies.base import (
    Order,
    OrderSide,
    Strategy,
    StrategyContext,
)


@dataclass
class RegimeState:
    """Lightweight snapshot the regime function may inspect."""

    bar: pd.Series
    history_close: list[float]


RegimeFn = Callable[[RegimeState], bool]


def sma_trend_filter(window: int) -> RegimeFn:
    """Regime is *on* when the latest close is above its trailing SMA.

    Until ``window`` bars of history are available the regime is *off*, so
    the wrapped strategy stays flat during the warm-up rather than placing
    blind entries on the first few bars.
    """
    if window < 2:
        raise ValueError("window must be at least 2")

    def _fn(state: RegimeState) -> bool:
        if len(state.history_close) < window:
            return False
        recent = state.history_close[-window:]
        sma = sum(recent) / window
        return float(state.bar["close"]) > sma

    return _fn


class RegimeFilteredStrategy(Strategy):
    """Wrap ``inner`` and only let it buy when the regime is on.

    Behaviour rules:

    * The inner strategy still sees every bar and runs its own logic in
      :meth:`on_bar`, so any internal indicators stay warm.
    * **Buy** orders from the inner strategy are forwarded only when the
      regime is on; otherwise they are silently dropped.
    * **Sell** orders from the inner strategy always pass through.
    * When the regime turns *off* and the wrapper still holds inventory
      (``ctx.position > 0``) we inject a synthetic full-position SELL on
      the next bar's open. We don't double-emit if the inner strategy
      already wanted to sell.
    """

    name = "regime_filtered"

    def __init__(
        self,
        inner: Strategy,
        regime_fn: RegimeFn | None = None,
        sma_window: int = 200,
    ) -> None:
        if regime_fn is None:
            regime_fn = sma_trend_filter(sma_window)
        self._inner = inner
        self._regime_fn = regime_fn
        self._history_close: list[float] = []
        self._sma_window = sma_window

    def reset(self) -> None:
        self._inner.reset()
        self._history_close = []

    def on_bar(self, bar: pd.Series, ctx: StrategyContext) -> list[Order]:
        regime_on = self._regime_fn(
            RegimeState(bar=bar, history_close=list(self._history_close))
        )
        self._history_close.append(float(bar["close"]))

        inner_orders = self._inner.on_bar(bar, ctx)
        forwarded: list[Order] = []
        already_selling = False
        for order in inner_orders:
            if order.side is OrderSide.BUY:
                if regime_on:
                    forwarded.append(order)
            else:
                forwarded.append(order)
                already_selling = True

        if not regime_on and ctx.position > 0 and not already_selling:
            forwarded.append(Order(side=OrderSide.SELL, position_fraction=1.0))

        return forwarded

    def describe(self) -> dict:
        return {
            "name": self.name,
            "inner": self._inner.describe(),
            "sma_window": self._sma_window,
        }
