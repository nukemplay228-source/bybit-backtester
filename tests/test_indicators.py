"""Unit tests for the vectorised indicators module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bybit_backtest.indicators import atr, donchian, ema, macd, rsi


def _make_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")


def test_ema_warmup_and_value() -> None:
    closes = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=_make_index(6))
    out = ema(closes, period=3)
    assert pd.isna(out.iloc[0])
    assert pd.isna(out.iloc[1])
    # adjust=False with alpha = 2/(period+1) = 0.5 and seed y_0 = x_0:
    #   y_0 = 1.0; y_1 = 0.5*2 + 0.5*1 = 1.5; y_2 = 0.5*3 + 0.5*1.5 = 2.25
    assert out.iloc[2] == pytest.approx(2.25, abs=1e-9)
    assert out.iloc[3] == pytest.approx(0.5 * 4 + 0.5 * 2.25, abs=1e-9)


def test_ema_invalid_period() -> None:
    closes = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        ema(closes, period=0)


def test_rsi_all_up_is_100() -> None:
    closes = pd.Series(np.arange(1.0, 50.0, 1.0), index=_make_index(49))
    out = rsi(closes, period=14)
    # Strictly increasing series -> no losses -> RSI saturates at 100.
    assert out.dropna().iloc[-1] == pytest.approx(100.0)


def test_rsi_invalid_period() -> None:
    with pytest.raises(ValueError):
        rsi(pd.Series([1.0, 2.0]), period=1)


def test_rsi_oscillating_in_range() -> None:
    n = 200
    t = np.arange(n, dtype="float64")
    closes = pd.Series(100.0 + 5.0 * np.sin(t / 5.0), index=_make_index(n))
    out = rsi(closes, period=14).dropna()
    assert ((out >= 0.0) & (out <= 100.0)).all()
    # An oscillator should produce both >70 and <30 readings somewhere.
    assert (out > 60).any()
    assert (out < 40).any()


def test_macd_invalid_params() -> None:
    closes = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        macd(closes, fast=26, slow=12, signal=9)
    with pytest.raises(ValueError):
        macd(closes, fast=0, slow=10, signal=9)


def test_macd_components_have_correct_relationship() -> None:
    n = 300
    t = np.arange(n, dtype="float64")
    closes = pd.Series(100.0 + 0.1 * t + 5.0 * np.sin(t / 20.0), index=_make_index(n))
    res = macd(closes, fast=12, slow=26, signal=9)
    # hist == macd - signal (where both defined)
    diff = (res.macd - res.signal) - res.hist
    assert diff.dropna().abs().max() < 1e-9


def test_atr_positive_and_finite() -> None:
    n = 100
    rng = np.random.default_rng(42)
    closes = 100.0 + rng.normal(0.0, 1.0, n).cumsum()
    high = closes + np.abs(rng.normal(0.0, 0.5, n))
    low = closes - np.abs(rng.normal(0.0, 0.5, n))
    idx = _make_index(n)
    out = atr(
        pd.Series(high, index=idx),
        pd.Series(low, index=idx),
        pd.Series(closes, index=idx),
        period=14,
    ).dropna()
    assert (out > 0).all()
    assert np.isfinite(out).all()


def test_donchian_uses_prior_window() -> None:
    closes = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=_make_index(6))
    chan = donchian(closes, closes, period=3)
    # At t=3 (zero-indexed), the prior 3 highs are [1,2,3] -> max 3.
    assert chan.high.iloc[3] == pytest.approx(3.0)
    assert chan.low.iloc[3] == pytest.approx(1.0)
    # At t=2 we don't yet have 3 prior bars -> NaN.
    assert pd.isna(chan.high.iloc[2])


def test_donchian_invalid_period() -> None:
    closes = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        donchian(closes, closes, period=1)
