"""Tests for the multi-signal predictor strategy and report builder."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bybit_backtest.backtest import run_backtest
from bybit_backtest.predict import build_report
from bybit_backtest.strategies import PredictorStrategy, build_strategy
from bybit_backtest.strategies.predictor import evaluate_signals, latest_signal_score


@pytest.fixture
def long_uptrend_bars() -> pd.DataFrame:
    """Long, mostly-trending series so all indicators have time to warm up."""
    n = 600
    rng = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    t = np.arange(n, dtype="float64")
    # Smooth uptrend with gentle wiggle so RSI doesn't permanently saturate.
    closes = 100.0 + 0.2 * t + 5.0 * np.sin(t / 30.0)
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": np.full(n, 1.0),
            "turnover": np.full(n, 100.0),
        },
        index=rng,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def long_downtrend_bars() -> pd.DataFrame:
    """A long, smooth down-trend — predictor must NOT go long here."""
    n = 600
    rng = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    t = np.arange(n, dtype="float64")
    closes = 200.0 - 0.2 * t + 4.0 * np.sin(t / 30.0)
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": np.full(n, 1.0),
            "turnover": np.full(n, 100.0),
        },
        index=rng,
    )
    df.index.name = "timestamp"
    return df


def test_predictor_invalid_params() -> None:
    with pytest.raises(ValueError):
        PredictorStrategy(fast_trend=200, slow_trend=50)
    with pytest.raises(ValueError):
        PredictorStrategy(rsi_buy_min=80, rsi_buy_max=70)
    with pytest.raises(ValueError):
        PredictorStrategy(min_score=0)
    with pytest.raises(ValueError):
        PredictorStrategy(atr_pct_min=0.1, atr_pct_max=0.05)


def test_predictor_registered() -> None:
    strat = build_strategy("predictor", {})
    assert isinstance(strat, PredictorStrategy)
    assert strat.describe()["name"] == "predictor"


def test_predictor_takes_long_in_uptrend(long_uptrend_bars: pd.DataFrame) -> None:
    strategy = PredictorStrategy(min_score=3)
    result = run_backtest(
        long_uptrend_bars,
        strategy,
        initial_cash=1000.0,
        fee_rate=0.0,
        slippage=0.0,
    )
    assert (result.trades["side"] == "BUY").sum() >= 1
    # In a clean up-trend with no fees the predictor should not lose money.
    assert result.equity.iloc[-1] >= result.equity.iloc[0] - 1e-6


def test_predictor_avoids_downtrend(long_downtrend_bars: pd.DataFrame) -> None:
    strategy = PredictorStrategy(min_score=4)
    result = run_backtest(
        long_downtrend_bars,
        strategy,
        initial_cash=1000.0,
        fee_rate=0.0,
        slippage=0.0,
    )
    # The predictor should refuse to trade a clean down-trend at min_score=4.
    assert (result.trades["side"] == "BUY").sum() == 0


def test_evaluate_signals_columns_and_score(long_uptrend_bars: pd.DataFrame) -> None:
    table = evaluate_signals(long_uptrend_bars)
    expected = {
        "trend_up",
        "macd_bullish",
        "rsi_strong",
        "breakout",
        "volatility_ok",
        "score",
        "max_score",
    }
    assert expected.issubset(set(table.columns))
    assert (table["score"] >= 0).all()
    assert (table["score"] <= 5).all()
    assert (table["max_score"] == 5).all()


def test_evaluate_signals_requires_columns() -> None:
    df = pd.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(ValueError):
        evaluate_signals(df)


def test_latest_signal_score_too_short() -> None:
    n = 50
    rng = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    closes = np.linspace(100.0, 110.0, n)
    df = pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes},
        index=rng,
    )
    with pytest.raises(ValueError):
        latest_signal_score(df)


def test_build_report_uptrend(long_uptrend_bars: pd.DataFrame) -> None:
    report = build_report(long_uptrend_bars, symbol="BTCUSDT", interval="60", min_score=3)
    assert report.symbol == "BTCUSDT"
    assert report.interval == "60"
    assert report.score.max_score == 5
    assert 0 <= report.score.score <= 5
    text = report.to_text()
    assert "Score" in text
    assert "Direction" in text
    # Round-trip the JSON dict.
    payload = json.loads(json.dumps(report.to_dict(), default=str))
    assert payload["symbol"] == "BTCUSDT"
    if report.direction == "LONG":
        assert report.suggested_stop is not None
        assert report.suggested_target is not None
        assert report.suggested_stop < report.close < report.suggested_target


def test_build_report_validates_min_score(long_uptrend_bars: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        build_report(long_uptrend_bars, symbol="X", interval="60", min_score=0)


def test_build_report_too_few_bars() -> None:
    n = 100
    rng = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    closes = np.linspace(100.0, 110.0, n)
    df = pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes},
        index=rng,
    )
    with pytest.raises(ValueError):
        build_report(df, symbol="X", interval="60")
