"""Vectorised technical indicators used by the predictor strategy and CLI.

These helpers operate on full :class:`pandas.Series` / :class:`pandas.DataFrame`
inputs and return aligned series. They are intentionally simple, dependency-free
(pandas/numpy only) and free of look-ahead bias: every value at index ``t``
only depends on data up to and including ``t``.

Included indicators:

* :func:`ema` -- exponential moving average (Wilder-compatible when ``adjust=False``).
* :func:`rsi` -- Wilder's Relative Strength Index.
* :func:`macd` -- MACD line, signal line and histogram.
* :func:`atr` -- Wilder's Average True Range.
* :func:`donchian` -- rolling highest-high / lowest-low channel
  (computed on the *prior* ``period`` bars, so the value at ``t`` excludes the
  current bar -- handy for breakout detection without look-ahead).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average using ``alpha = 2 / (period + 1)``."""
    if period <= 0:
        raise ValueError("period must be positive")
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index.

    Returns NaN until ``period`` price changes are available. Values are
    bounded between 0 and 100; an all-flat input returns 100 (no losses).
    """
    if period < 2:
        raise ValueError("period must be at least 2")
    change = series.diff()
    gain = change.clip(lower=0.0)
    loss = (-change).clip(lower=0.0)
    # Wilder's smoothing == EMA with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss is exactly zero, RSI is conventionally 100.
    out = out.where(avg_loss != 0.0, 100.0)
    return out


@dataclass(frozen=True)
class MACDResult:
    """Container holding the three components of a MACD reading."""

    macd: pd.Series
    signal: pd.Series
    hist: pd.Series


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> MACDResult:
    """Classic MACD: ``EMA(fast) - EMA(slow)`` with an EMA signal line."""
    if fast <= 0 or slow <= 0 or signal <= 0:
        raise ValueError("fast, slow and signal periods must be positive")
    if fast >= slow:
        raise ValueError("fast period must be strictly smaller than slow period")
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line.dropna(), signal).reindex(series.index)
    hist = macd_line - signal_line
    return MACDResult(macd=macd_line, signal=signal_line, hist=hist)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Wilder's Average True Range.

    True Range is ``max(high - low, |high - prev_close|, |low - prev_close|)``;
    ATR is the Wilder-smoothed average of TR over ``period`` bars.
    """
    if period < 2:
        raise ValueError("period must be at least 2")
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


@dataclass(frozen=True)
class DonchianChannel:
    """Highest high and lowest low over the *prior* ``period`` bars."""

    high: pd.Series
    low: pd.Series


def donchian(high: pd.Series, low: pd.Series, period: int = 20) -> DonchianChannel:
    """Rolling Donchian channel computed on the previous ``period`` bars.

    The channel at index ``t`` looks at bars ``[t-period, t-1]`` -- it does
    NOT include the current bar. This is what you want for breakout
    detection: "is today's close above the highest of the last N closes?".
    """
    if period < 2:
        raise ValueError("period must be at least 2")
    return DonchianChannel(
        high=high.shift(1).rolling(period, min_periods=period).max(),
        low=low.shift(1).rolling(period, min_periods=period).min(),
    )
