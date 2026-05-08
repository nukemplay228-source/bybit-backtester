"""Spot grid strategy (long-only).

Splits ``[low, high]`` into ``levels`` equal-spaced price levels. The grid is
divided into uniform "slots" — each slot is bought when price crosses below
its level and sold when price crosses back above the next level up. This is
a simple educational implementation: it ignores partial fills, treats every
level as a market order on the next bar's open, and keeps grid state in
memory rather than as on-exchange limit orders.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bybit_backtest.strategies.base import Order, OrderSide, Strategy, StrategyContext


@dataclass
class _Slot:
    """A single grid slot, defined by a buy level and a sell level above it."""

    buy_level: float
    sell_level: float
    holding: bool = False
    qty: float = 0.0


class SpotGridStrategy(Strategy):
    name = "grid"

    def __init__(
        self,
        low: float,
        high: float,
        levels: int = 20,
        capital_fraction: float = 1.0,
    ) -> None:
        if low <= 0 or high <= 0:
            raise ValueError("low and high must be positive")
        if low >= high:
            raise ValueError("low must be strictly less than high")
        if levels < 2:
            raise ValueError("levels must be at least 2")
        if not 0 < capital_fraction <= 1:
            raise ValueError("capital_fraction must be in (0, 1]")
        self.low = float(low)
        self.high = float(high)
        self.levels = int(levels)
        self.capital_fraction = float(capital_fraction)
        # Build slots between adjacent levels.
        step = (self.high - self.low) / (self.levels - 1)
        self._slots: list[_Slot] = [
            _Slot(
                buy_level=self.low + i * step,
                sell_level=self.low + (i + 1) * step,
            )
            for i in range(self.levels - 1)
        ]
        self._allocated = False

    def reset(self) -> None:
        for slot in self._slots:
            slot.holding = False
            slot.qty = 0.0
        self._allocated = False

    def describe(self) -> dict:
        return {
            "name": self.name,
            "low": self.low,
            "high": self.high,
            "levels": self.levels,
            "capital_fraction": self.capital_fraction,
        }

    def on_bar(self, bar: pd.Series, ctx: StrategyContext) -> list[Order]:
        close = float(bar["close"])
        orders: list[Order] = []
        n_slots = len(self._slots)

        # Sells happen first: any holding slot whose sell_level was crossed.
        for slot in self._slots:
            if slot.holding and close >= slot.sell_level and ctx.position > 0:
                fraction = slot.qty / max(ctx.position, 1e-12)
                fraction = min(max(fraction, 0.0), 1.0)
                if fraction > 0:
                    orders.append(
                        Order(side=OrderSide.SELL, position_fraction=fraction)
                    )
                slot.holding = False
                slot.qty = 0.0

        # Buys: any non-holding slot whose buy_level is at or below price.
        for slot in self._slots:
            if not slot.holding and close <= slot.buy_level and ctx.cash > 0:
                allowance = (
                    self.capital_fraction * ctx.equity / n_slots
                )
                cash_fraction = min(allowance / max(ctx.cash, 1e-12), 1.0)
                if cash_fraction > 0:
                    orders.append(
                        Order(side=OrderSide.BUY, cash_fraction=cash_fraction)
                    )
                # Estimate qty so we know how much to sell when level is hit.
                qty_estimate = allowance / close
                slot.qty = qty_estimate
                slot.holding = True

        return orders
