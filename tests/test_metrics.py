"""Tests for performance metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from bybit_backtest.metrics import compute_report, max_drawdown


def _series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="1h", tz="UTC")
    return pd.Series(values, index=idx, name="equity")


def test_max_drawdown_simple() -> None:
    s = _series([100, 120, 90, 80, 100])
    # Peak 120 -> trough 80 = -33.33%
    assert max_drawdown(s) == pytest.approx((120 - 80) / 120, rel=1e-6)


def test_max_drawdown_no_drawdown() -> None:
    s = _series([100, 110, 120, 130])
    assert max_drawdown(s) == pytest.approx(0.0, abs=1e-9)


def test_compute_report_total_return() -> None:
    equity = _series([100.0, 110.0, 121.0])
    trades = pd.DataFrame(
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
    report = compute_report(equity, trades, interval="60")
    assert report.total_return_pct == pytest.approx(21.0, rel=1e-6)
    assert report.num_trades == 0


def test_compute_report_with_round_trips() -> None:
    equity = _series([100.0, 105.0, 95.0, 110.0])
    trades = pd.DataFrame(
        [
            {
                "timestamp": equity.index[0],
                "side": "BUY",
                "price": 100.0,
                "quantity": 1.0,
                "fee": 0.0,
                "cash_after": 0.0,
                "position_after": 1.0,
                "equity_after": 100.0,
            },
            {
                "timestamp": equity.index[1],
                "side": "SELL",
                "price": 105.0,
                "quantity": 1.0,
                "fee": 0.0,
                "cash_after": 105.0,
                "position_after": 0.0,
                "equity_after": 105.0,
            },
            {
                "timestamp": equity.index[2],
                "side": "BUY",
                "price": 95.0,
                "quantity": 1.0,
                "fee": 0.0,
                "cash_after": 0.0,
                "position_after": 1.0,
                "equity_after": 95.0,
            },
            {
                "timestamp": equity.index[3],
                "side": "SELL",
                "price": 110.0,
                "quantity": 1.0,
                "fee": 0.0,
                "cash_after": 110.0,
                "position_after": 0.0,
                "equity_after": 110.0,
            },
        ]
    )
    report = compute_report(equity, trades, interval="60")
    assert report.num_trades == 2
    assert report.win_rate_pct == pytest.approx(100.0)
    assert report.avg_trade_pct > 0
    assert report.profit_factor > 1.0


def test_compute_report_handles_empty_equity() -> None:
    report = compute_report(pd.Series(dtype="float64"), pd.DataFrame(), interval="60")
    assert report.num_trades == 0
    assert report.total_return_pct == 0.0
