"""Portfolio bookkeeping for the long-only spot backtester."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


@dataclass
class Trade:
    """A single round-trip leg (one buy or one sell)."""

    timestamp: datetime
    side: str  # "BUY" or "SELL"
    price: float
    quantity: float
    fee: float
    cash_after: float
    position_after: float
    equity_after: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "side": self.side,
            "price": self.price,
            "quantity": self.quantity,
            "fee": self.fee,
            "cash_after": self.cash_after,
            "position_after": self.position_after,
            "equity_after": self.equity_after,
        }


@dataclass
class Portfolio:
    """Tracks cash, position, and equity over the course of a backtest.

    The backtester only supports long-only spot trades — short selling and
    margin are intentionally out of scope to keep results easy to reason
    about for a beginner.
    """

    initial_cash: float
    fee_rate: float = 0.001  # 0.1% per trade, matches Bybit spot taker default
    cash: float = field(init=False)
    position: float = field(init=False, default=0.0)
    avg_entry_price: float = field(init=False, default=0.0)
    trades: list[Trade] = field(default_factory=list, init=False)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.fee_rate < 0:
            raise ValueError("fee_rate must be non-negative")
        self.cash = float(self.initial_cash)

    # ------------------------------------------------------------------
    # Equity helpers
    # ------------------------------------------------------------------
    def equity(self, mark_price: float) -> float:
        return self.cash + self.position * mark_price

    def record_equity(self, timestamp: datetime, mark_price: float) -> None:
        self.equity_curve.append((timestamp, self.equity(mark_price)))

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------
    def buy(self, timestamp: datetime, price: float, cash_to_spend: float) -> Trade | None:
        """Spend ``cash_to_spend`` to buy at ``price``. Fee is deducted from cash."""
        if price <= 0:
            raise ValueError("price must be positive")
        if cash_to_spend <= 0:
            return None
        cash_to_spend = min(cash_to_spend, self.cash)
        if cash_to_spend <= 0:
            return None

        # We pay fee out of the same cash bucket: gross = cash_to_spend
        # quantity * price + fee = cash_to_spend, fee = quantity * price * fee_rate
        # => quantity = cash_to_spend / (price * (1 + fee_rate))
        quantity = cash_to_spend / (price * (1.0 + self.fee_rate))
        if quantity <= 0:
            return None
        notional = quantity * price
        fee = notional * self.fee_rate
        spent = notional + fee

        # Update average entry price (weighted by quantity).
        if self.position > 0:
            self.avg_entry_price = (
                self.avg_entry_price * self.position + price * quantity
            ) / (self.position + quantity)
        else:
            self.avg_entry_price = price

        self.cash -= spent
        self.position += quantity

        trade = Trade(
            timestamp=timestamp,
            side="BUY",
            price=price,
            quantity=quantity,
            fee=fee,
            cash_after=self.cash,
            position_after=self.position,
            equity_after=self.equity(price),
        )
        self.trades.append(trade)
        return trade

    def sell(self, timestamp: datetime, price: float, quantity: float) -> Trade | None:
        if price <= 0:
            raise ValueError("price must be positive")
        if quantity <= 0:
            return None
        quantity = min(quantity, self.position)
        if quantity <= 0:
            return None
        notional = quantity * price
        fee = notional * self.fee_rate
        proceeds = notional - fee
        self.cash += proceeds
        self.position -= quantity
        if self.position <= 1e-12:
            self.position = 0.0
            self.avg_entry_price = 0.0

        trade = Trade(
            timestamp=timestamp,
            side="SELL",
            price=price,
            quantity=quantity,
            fee=fee,
            cash_after=self.cash,
            position_after=self.position,
            equity_after=self.equity(price),
        )
        self.trades.append(trade)
        return trade

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype="float64")
        idx = pd.DatetimeIndex([t for t, _ in self.equity_curve], tz="UTC")
        values = [v for _, v in self.equity_curve]
        return pd.Series(values, index=idx, name="equity")

    def trades_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "side",
                    "price",
                    "quantity",
                    "fee",
                    "cash_after",
                    "position_after",
                    "equity_after",
                ]
            )
        return pd.DataFrame([t.to_dict() for t in self.trades])
