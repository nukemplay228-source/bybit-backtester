from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bybit_backtest.backtest import run_backtest
from bybit_backtest.regime import (
    RegimeFilteredStrategy,
    RegimeState,
    sma_trend_filter,
)
from bybit_backtest.strategies.base import (
    Order,
    OrderSide,
    Strategy,
    StrategyContext,
)
from bybit_backtest.strategies.sma_cross import SMACrossStrategy


def _make_bar(close: float) -> pd.Series:
    return pd.Series({"open": close, "high": close, "low": close, "close": close, "volume": 1.0})


def _make_ctx(*, position: float = 0.0) -> StrategyContext:
    return StrategyContext(cash=1000.0, position=position, equity=1000.0, avg_entry_price=0.0)


class _AlwaysBuy(Strategy):
    """Helper inner strategy that emits a single full-cash BUY every bar."""

    name = "always_buy"

    def on_bar(self, bar: pd.Series, ctx: StrategyContext) -> list[Order]:
        return [Order(side=OrderSide.BUY, cash_fraction=1.0)]


class _ScriptedStrategy(Strategy):
    """Helper inner strategy that replays a pre-recorded list of orders."""

    name = "scripted"

    def __init__(self, orders_per_bar: list[list[Order]]):
        self._orders = orders_per_bar
        self._i = 0

    def reset(self) -> None:
        self._i = 0

    def on_bar(self, bar: pd.Series, ctx: StrategyContext) -> list[Order]:
        out = self._orders[self._i] if self._i < len(self._orders) else []
        self._i += 1
        return out


def test_sma_trend_filter_off_during_warmup_then_compares_close_to_sma() -> None:
    fn = sma_trend_filter(window=5)
    closes: list[float] = []
    for c in [10.0, 11.0, 12.0]:
        assert fn(RegimeState(bar=_make_bar(c), history_close=list(closes))) is False
        closes.append(c)

    closes_above_sma = [10.0, 11.0, 12.0, 13.0, 14.0]
    assert fn(RegimeState(bar=_make_bar(15.0), history_close=closes_above_sma)) is True

    closes_below_sma = [20.0, 19.0, 18.0, 17.0, 16.0]
    assert fn(RegimeState(bar=_make_bar(10.0), history_close=closes_below_sma)) is False


def test_regime_blocks_buy_signals_when_regime_off() -> None:
    bear_closes = [100.0, 95.0, 90.0, 85.0, 80.0, 75.0]
    wrapped = RegimeFilteredStrategy(
        inner=_AlwaysBuy(), regime_fn=sma_trend_filter(window=3)
    )
    wrapped.reset()
    ctx = _make_ctx()
    for close in bear_closes:
        out = wrapped.on_bar(_make_bar(close), ctx)
        assert out == []


def test_regime_lets_buy_through_during_warmup_only_when_regime_on() -> None:
    closes = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
    wrapped = RegimeFilteredStrategy(
        inner=_AlwaysBuy(), regime_fn=sma_trend_filter(window=3)
    )
    wrapped.reset()
    ctx = _make_ctx()
    seen_buy = False
    for close in closes:
        out = wrapped.on_bar(_make_bar(close), ctx)
        if out:
            assert all(o.side is OrderSide.BUY for o in out)
            seen_buy = True
    assert seen_buy


def test_regime_force_exits_when_position_open_and_regime_flips_off() -> None:
    history = [110.0, 108.0, 106.0]
    wrapped = RegimeFilteredStrategy(
        inner=_ScriptedStrategy([[]]), regime_fn=sma_trend_filter(window=3)
    )
    wrapped.reset()
    for close in history:
        wrapped.on_bar(_make_bar(close), _make_ctx())

    ctx_with_position = _make_ctx(position=1.0)
    out = wrapped.on_bar(_make_bar(50.0), ctx_with_position)
    assert len(out) == 1
    assert out[0].side is OrderSide.SELL
    assert out[0].position_fraction == pytest.approx(1.0)


def test_regime_does_not_double_emit_when_inner_already_sells() -> None:
    history = [110.0, 108.0, 106.0]
    inner_sell = [Order(side=OrderSide.SELL, position_fraction=0.5)]
    wrapped = RegimeFilteredStrategy(
        inner=_ScriptedStrategy([inner_sell, inner_sell]),
        regime_fn=sma_trend_filter(window=3),
    )
    wrapped.reset()
    for close in history:
        wrapped.on_bar(_make_bar(close), _make_ctx())
    out = wrapped.on_bar(_make_bar(50.0), _make_ctx(position=1.0))
    sell_orders = [o for o in out if o.side is OrderSide.SELL]
    assert len(sell_orders) == 1


def test_regime_filtered_sma_outperforms_or_matches_in_bear_then_bull() -> None:
    """Functional integration test against the real backtest engine.

    The synthetic 2-year price path is a deep bear followed by a strong bull;
    the regime filter should hold the wrapped SMA flat through the bear and
    let it ride the bull, so its drawdown must be no worse than the raw SMA.
    """
    n = 800
    bear = np.linspace(100.0, 30.0, n // 2)
    bull = np.linspace(30.0, 200.0, n - n // 2)
    closes = np.concatenate([bear, bull])
    idx = pd.date_range("2023-01-01", periods=n, freq="6h", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.001,
            "low": closes * 0.999,
            "close": closes,
            "volume": np.full(n, 1.0),
        },
        index=idx,
    )

    raw = run_backtest(
        bars,
        SMACrossStrategy(fast=5, slow=20),
        initial_cash=10_000.0,
        fee_rate=0.0,
        slippage=0.0,
    )
    filtered = run_backtest(
        bars,
        RegimeFilteredStrategy(
            inner=SMACrossStrategy(fast=5, slow=20),
            regime_fn=sma_trend_filter(window=200),
        ),
        initial_cash=10_000.0,
        fee_rate=0.0,
        slippage=0.0,
    )

    assert filtered.report.max_drawdown_pct <= raw.report.max_drawdown_pct + 1e-6
