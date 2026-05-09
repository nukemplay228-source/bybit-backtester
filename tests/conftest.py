"""Shared fixtures for the bybit-backtester test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def trending_bars() -> pd.DataFrame:
    """100 hourly bars in a clean uptrend (then sideways)."""
    rng = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    base = np.linspace(100.0, 200.0, 60).tolist() + [200.0] * 40
    base_arr = np.array(base, dtype="float64")
    df = pd.DataFrame(
        {
            "open": base_arr,
            "high": base_arr * 1.005,
            "low": base_arr * 0.995,
            "close": base_arr,
            "volume": np.full(100, 10.0),
            "turnover": np.full(100, 1000.0),
        },
        index=rng,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def crossing_bars() -> pd.DataFrame:
    """Bars that produce a clear SMA-fast crossing the SMA-slow upward."""
    rng = pd.date_range("2024-01-01", periods=120, freq="1h", tz="UTC")
    # First half: down trend, second half: up trend
    first = np.linspace(200.0, 100.0, 60)
    second = np.linspace(100.0, 250.0, 60)
    closes = np.concatenate([first, second])
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.002,
            "low": closes * 0.998,
            "close": closes,
            "volume": np.full(120, 1.0),
            "turnover": np.full(120, 100.0),
        },
        index=rng,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def oscillating_bars() -> pd.DataFrame:
    """Sinusoidal price action — useful for grid / mean reversion strategies."""
    rng = pd.date_range("2024-01-01", periods=240, freq="1h", tz="UTC")
    t = np.arange(240, dtype="float64")
    closes = 100.0 + 10.0 * np.sin(t / 8.0)
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": np.full(240, 1.0),
            "turnover": np.full(240, 100.0),
        },
        index=rng,
    )
    df.index.name = "timestamp"
    return df
