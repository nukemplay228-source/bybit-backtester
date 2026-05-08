"""Tests for the Portfolio bookkeeping."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bybit_backtest.portfolio import Portfolio


def _ts(offset: int = 0) -> datetime:
    return datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc).replace(hour=offset)


def test_initial_state() -> None:
    p = Portfolio(initial_cash=1000.0, fee_rate=0.001)
    assert p.cash == 1000.0
    assert p.position == 0.0
    assert p.equity(mark_price=100.0) == 1000.0


def test_buy_uses_full_cash_minus_fee() -> None:
    p = Portfolio(initial_cash=1000.0, fee_rate=0.001)
    trade = p.buy(_ts(1), price=100.0, cash_to_spend=1000.0)

    assert trade is not None
    assert trade.side == "BUY"
    # quantity = 1000 / (100 * 1.001) ≈ 9.990009...
    assert trade.quantity == pytest.approx(1000.0 / (100.0 * 1.001), rel=1e-9)
    notional = trade.quantity * 100.0
    assert trade.fee == pytest.approx(notional * 0.001, rel=1e-9)
    assert p.cash == pytest.approx(0.0, abs=1e-9)
    assert p.position == pytest.approx(trade.quantity, rel=1e-9)
    assert p.avg_entry_price == 100.0


def test_sell_closes_full_position() -> None:
    p = Portfolio(initial_cash=1000.0, fee_rate=0.001)
    p.buy(_ts(1), price=100.0, cash_to_spend=1000.0)
    qty = p.position
    trade = p.sell(_ts(2), price=110.0, quantity=qty)

    assert trade is not None
    assert p.position == 0.0
    assert p.avg_entry_price == 0.0
    # 10% rise, 0.1% fee both ways → net pnl is positive but less than 10%.
    assert p.cash > 1000.0
    assert p.cash < 1100.0


def test_zero_or_negative_input_returns_none() -> None:
    p = Portfolio(initial_cash=1000.0)
    assert p.buy(_ts(1), price=100.0, cash_to_spend=0.0) is None
    assert p.sell(_ts(1), price=100.0, quantity=0.0) is None


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        Portfolio(initial_cash=0)
    p = Portfolio(initial_cash=1000.0)
    with pytest.raises(ValueError):
        p.buy(_ts(1), price=-1.0, cash_to_spend=10.0)
    with pytest.raises(ValueError):
        p.sell(_ts(1), price=0.0, quantity=10.0)


def test_partial_buys_update_avg_entry_price() -> None:
    p = Portfolio(initial_cash=1000.0, fee_rate=0.0)
    p.buy(_ts(1), price=100.0, cash_to_spend=400.0)
    p.buy(_ts(2), price=200.0, cash_to_spend=400.0)
    # qty1 = 4 at price 100, qty2 = 2 at price 200, weighted avg = 800/6 = 133.33
    assert p.avg_entry_price == pytest.approx((4 * 100 + 2 * 200) / 6, rel=1e-9)


def test_equity_curve_recorded() -> None:
    p = Portfolio(initial_cash=500.0, fee_rate=0.0)
    p.record_equity(_ts(0), 100.0)
    p.buy(_ts(1), price=100.0, cash_to_spend=500.0)
    p.record_equity(_ts(2), 110.0)
    series = p.equity_series()
    assert len(series) == 2
    assert series.iloc[0] == 500.0
    assert series.iloc[-1] == pytest.approx(550.0, rel=1e-9)
