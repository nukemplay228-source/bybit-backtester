"""End-to-end tests for the backtest engine."""

from __future__ import annotations

import pandas as pd
import pytest

from bybit_backtest.backtest import run_backtest
from bybit_backtest.strategies.base import Order, OrderSide, Strategy, StrategyContext


class _BuyAndHoldStrategy(Strategy):
    name = "buy_and_hold"

    def __init__(self) -> None:
        self._bought = False

    def reset(self) -> None:
        self._bought = False

    def on_bar(self, bar: pd.Series, ctx: StrategyContext) -> list[Order]:
        if not self._bought and ctx.cash > 0:
            self._bought = True
            return [Order(side=OrderSide.BUY, cash_fraction=1.0)]
        return []


def test_engine_rejects_missing_columns() -> None:
    bars = pd.DataFrame(
        {"open": [1.0, 2.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC"),
    )
    with pytest.raises(ValueError):
        run_backtest(bars, _BuyAndHoldStrategy())


def test_engine_rejects_naive_index() -> None:
    bars = pd.DataFrame(
        {col: [1.0, 2.0] for col in ("open", "high", "low", "close")},
        index=pd.date_range("2024-01-01", periods=2, freq="1h"),
    )
    with pytest.raises(ValueError):
        run_backtest(bars, _BuyAndHoldStrategy())


def test_buy_and_hold_tracks_close(trending_bars: pd.DataFrame) -> None:
    result = run_backtest(
        trending_bars,
        _BuyAndHoldStrategy(),
        initial_cash=1000.0,
        fee_rate=0.0,
        slippage=0.0,
    )
    # Order is queued on bar 0's close, executed on bar 1's open. After that,
    # equity should track the close price exactly (no fees, no slippage).
    assert result.equity.iloc[-1] > result.equity.iloc[0]
    final_equity_ratio = result.equity.iloc[-1] / trending_bars["close"].iloc[-1]
    initial_equity_ratio = result.equity.iloc[1] / trending_bars["close"].iloc[1]
    assert final_equity_ratio == pytest.approx(initial_equity_ratio, rel=1e-6)


def test_no_lookahead_bias(trending_bars: pd.DataFrame) -> None:
    """A strategy that decides on close should not transact at the same bar."""
    result = run_backtest(
        trending_bars,
        _BuyAndHoldStrategy(),
        initial_cash=1000.0,
        fee_rate=0.0,
        slippage=0.0,
    )
    first_trade_ts = result.trades["timestamp"].iloc[0]
    # The buy decision is made on the first bar's close, but the trade
    # should be executed at the *second* bar's open.
    assert pd.Timestamp(first_trade_ts) == trending_bars.index[1]
