"""Strategy interface used by the backtester.

A strategy receives one bar at a time via :meth:`Strategy.on_bar` together
with a :class:`StrategyContext` describing the current portfolio state.
It returns zero or more :class:`Order` instructions that the engine will
execute on the *next* bar's open. Decisions are made on close-of-bar; this
avoids look-ahead bias.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Order:
    """A single market-on-next-open instruction.

    For BUY orders ``cash_fraction`` represents the fraction of *available
    cash* to spend (0 < fraction <= 1). For SELL orders ``position_fraction``
    represents the fraction of the current position to liquidate.
    """

    side: OrderSide
    cash_fraction: float = 0.0
    position_fraction: float = 0.0


@dataclass
class StrategyContext:
    """Snapshot of portfolio state passed to the strategy each bar."""

    cash: float
    position: float
    equity: float
    avg_entry_price: float


class Strategy(ABC):
    """Abstract base class. Subclass and implement :meth:`on_bar`."""

    name: str = "abstract"

    @abstractmethod
    def on_bar(self, bar: pd.Series, ctx: StrategyContext) -> list[Order]:
        """Return a list of orders to execute on the next bar's open."""

    def reset(self) -> None:
        """Reset internal state. Called once at the start of each backtest run."""
        return None

    def describe(self) -> dict:
        """Return a JSON-serializable summary of the strategy parameters."""
        return {"name": self.name}
