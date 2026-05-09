"""Generate a live "where might this coin go?" report for a Bybit symbol.

This module wraps the multi-signal indicator evaluation from
:mod:`bybit_backtest.strategies.predictor` into a single report object
suitable for printing on the command line.

> [!CAUTION]
> The output is **not** a guaranteed prediction. It is a snapshot of how a
> handful of classical indicators line up on the most recent closed bar. Any
> action you take on it is at your own risk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from bybit_backtest.strategies.predictor import SignalScore, evaluate_signals


@dataclass(frozen=True)
class PredictionReport:
    """Human-readable summary of the most recent bar's confluence signal."""

    symbol: str
    interval: str
    timestamp: pd.Timestamp
    close: float
    fast_ema: float
    slow_ema: float
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    atr: float
    atr_pct: float
    donchian_high: float
    donchian_low: float
    score: SignalScore
    direction: str
    suggested_stop: float | None
    suggested_target: float | None
    atr_stop_mult: float
    atr_target_mult: float

    def to_dict(self) -> dict:
        out = asdict(self)
        out["timestamp"] = self.timestamp.isoformat()
        return out

    def to_text(self) -> str:
        lines = [
            f"Symbol      : {self.symbol}  (interval={self.interval})",
            f"Last close  : {self.close:.6f}  @ {self.timestamp.isoformat()}",
            "",
            "Indicators:",
            f"  EMA fast / slow   : {self.fast_ema:.6f} / {self.slow_ema:.6f}",
            f"  RSI               : {self.rsi:.2f}",
            f"  MACD / signal     : {self.macd:.6f} / {self.macd_signal:.6f}  (hist={self.macd_hist:+.6f})",
            f"  ATR ({self.atr_pct * 100:.2f}% of price): {self.atr:.6f}",
            f"  Donchian hi / lo  : {self.donchian_high:.6f} / {self.donchian_low:.6f}",
            "",
            "Signals:",
            f"  trend_up      : {self.score.trend_up}",
            f"  macd_bullish  : {self.score.macd_bullish}",
            f"  rsi_strong    : {self.score.rsi_strong}",
            f"  breakout      : {self.score.breakout}",
            f"  volatility_ok : {self.score.volatility_ok}",
            "",
            f"Score       : {self.score.score} / {self.score.max_score}  "
            f"(confidence={self.score.confidence * 100:.0f}%)",
            f"Direction   : {self.direction}",
        ]
        if self.suggested_stop is not None and self.suggested_target is not None:
            lines.extend(
                [
                    f"Suggested stop   : {self.suggested_stop:.6f} "
                    f"(-{self.atr_stop_mult:.1f}xATR from close)",
                    f"Suggested target : {self.suggested_target:.6f} "
                    f"(+{self.atr_target_mult:.1f}xATR from close)",
                ]
            )
        return "\n".join(lines)


def _direction_for(score: SignalScore, min_score: int) -> str:
    if score.score >= min_score:
        return "LONG"
    if score.score <= max(0, score.max_score - min_score):
        return "AVOID"
    return "NEUTRAL"


def build_report(
    bars: pd.DataFrame,
    *,
    symbol: str,
    interval: str,
    min_score: int = 4,
    atr_stop_mult: float = 2.0,
    atr_target_mult: float = 3.0,
    fast_trend: int = 50,
    slow_trend: int = 200,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    rsi_period: int = 14,
    rsi_buy_min: float = 50.0,
    rsi_buy_max: float = 70.0,
    donchian_period: int = 20,
    atr_period: int = 14,
    atr_pct_min: float = 0.002,
    atr_pct_max: float = 0.10,
) -> PredictionReport:
    """Compute indicators on ``bars`` and summarise the most recent closed bar.

    ``bars`` must be the same OHLCV DataFrame the backtester uses (UTC index,
    ``open/high/low/close`` columns). Raises :class:`ValueError` if it is too
    short for the configured indicator windows.
    """
    if bars.empty:
        raise ValueError("bars is empty")
    if not 1 <= min_score <= 5:
        raise ValueError("min_score must be between 1 and 5")
    if atr_stop_mult <= 0 or atr_target_mult <= 0:
        raise ValueError("ATR multipliers must be positive")

    table = evaluate_signals(
        bars,
        fast_trend=fast_trend,
        slow_trend=slow_trend,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        rsi_period=rsi_period,
        rsi_buy_min=rsi_buy_min,
        rsi_buy_max=rsi_buy_max,
        donchian_period=donchian_period,
        atr_period=atr_period,
        atr_pct_min=atr_pct_min,
        atr_pct_max=atr_pct_max,
    )
    last = table.iloc[-1]
    if last[["fast_ema", "slow_ema", "macd", "macd_signal", "atr"]].isna().any():
        raise ValueError(
            "not enough bars for the configured indicators "
            f"(need at least {slow_trend + macd_signal} bars, got {len(bars)})"
        )

    score = SignalScore(
        trend_up=bool(last["trend_up"]),
        macd_bullish=bool(last["macd_bullish"]),
        rsi_strong=bool(last["rsi_strong"]),
        breakout=bool(last["breakout"]),
        volatility_ok=bool(last["volatility_ok"]),
        score=int(last["score"]),
        max_score=int(last["max_score"]),
    )

    close = float(last["close"])
    atr_value = float(last["atr"])
    direction = _direction_for(score, min_score)
    if direction == "LONG":
        stop = close - atr_stop_mult * atr_value
        target = close + atr_target_mult * atr_value
    else:
        stop = None
        target = None

    return PredictionReport(
        symbol=symbol,
        interval=interval,
        timestamp=bars.index[-1],
        close=close,
        fast_ema=float(last["fast_ema"]),
        slow_ema=float(last["slow_ema"]),
        rsi=float(last["rsi"]),
        macd=float(last["macd"]),
        macd_signal=float(last["macd_signal"]),
        macd_hist=float(last["macd_hist"]),
        atr=atr_value,
        atr_pct=float(last["atr_pct"]),
        donchian_high=float(last["donchian_high"]),
        donchian_low=float(last["donchian_low"]),
        score=score,
        direction=direction,
        suggested_stop=stop,
        suggested_target=target,
        atr_stop_mult=atr_stop_mult,
        atr_target_mult=atr_target_mult,
    )
