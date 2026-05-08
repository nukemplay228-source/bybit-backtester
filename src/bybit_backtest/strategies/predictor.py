"""Multi-signal "predictor" strategy (long-only).

This strategy combines several classical technical signals -- a long-term EMA
trend filter, MACD momentum, RSI strength, a Donchian breakout confirmation
and an ATR volatility regime -- and only enters a long position when *enough
of them agree*. Entries get an ATR-based hard stop and an EMA-based trailing
exit.

The idea is **confluence**: any single indicator gives a noisy signal, but
asking several independent indicators to line up filters out a large share of
false positives. This does not magically beat the market -- in choppy or
trending-against-you regimes it will still lose -- but it is a more robust
baseline than a single moving-average crossover.

> [!CAUTION]
> No technical strategy "predicts" prices with high probability. Backtest
> every parameter set on out-of-sample data and assume realistic fees and
> slippage before risking real money.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pandas as pd

from bybit_backtest.indicators import atr, donchian, ema, macd, rsi
from bybit_backtest.strategies.base import Order, OrderSide, Strategy, StrategyContext


@dataclass(frozen=True)
class SignalScore:
    """Per-signal vote and aggregate confidence for a single bar."""

    trend_up: bool
    macd_bullish: bool
    rsi_strong: bool
    breakout: bool
    volatility_ok: bool
    score: int
    max_score: int

    @property
    def confidence(self) -> float:
        if self.max_score <= 0:
            return 0.0
        return self.score / self.max_score


def evaluate_signals(
    bars: pd.DataFrame,
    *,
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
) -> pd.DataFrame:
    """Compute the full per-bar signal table used by the predictor.

    Returns a DataFrame indexed like ``bars`` with one boolean column per
    signal plus integer ``score`` and ``max_score`` columns. Rows where any
    indicator is still warming up are NaN/False.
    """
    if bars.empty:
        raise ValueError("bars must contain at least one row")
    required = {"open", "high", "low", "close"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars is missing required columns: {sorted(missing)}")

    close = bars["close"].astype("float64")
    high = bars["high"].astype("float64")
    low = bars["low"].astype("float64")

    fast_ema = ema(close, fast_trend)
    slow_ema = ema(close, slow_trend)
    rsi_series = rsi(close, rsi_period)
    macd_res = macd(close, macd_fast, macd_slow, macd_signal)
    atr_series = atr(high, low, close, atr_period)
    channel = donchian(high, low, donchian_period)

    atr_pct = atr_series / close

    trend_up = (fast_ema > slow_ema).fillna(False)
    macd_bullish = ((macd_res.macd > macd_res.signal) & (macd_res.hist > 0)).fillna(False)
    rsi_strong = rsi_series.between(rsi_buy_min, rsi_buy_max).fillna(False)
    breakout = (close > channel.high).fillna(False)
    volatility_ok = atr_pct.between(atr_pct_min, atr_pct_max).fillna(False)

    table = pd.DataFrame(
        {
            "close": close,
            "fast_ema": fast_ema,
            "slow_ema": slow_ema,
            "rsi": rsi_series,
            "macd": macd_res.macd,
            "macd_signal": macd_res.signal,
            "macd_hist": macd_res.hist,
            "atr": atr_series,
            "atr_pct": atr_pct,
            "donchian_high": channel.high,
            "donchian_low": channel.low,
            "trend_up": trend_up,
            "macd_bullish": macd_bullish,
            "rsi_strong": rsi_strong,
            "breakout": breakout,
            "volatility_ok": volatility_ok,
        }
    )
    table["score"] = (
        table[["trend_up", "macd_bullish", "rsi_strong", "breakout", "volatility_ok"]]
        .astype("int64")
        .sum(axis=1)
    )
    table["max_score"] = 5
    return table


def _last_score(row: pd.Series) -> SignalScore:
    return SignalScore(
        trend_up=bool(row["trend_up"]),
        macd_bullish=bool(row["macd_bullish"]),
        rsi_strong=bool(row["rsi_strong"]),
        breakout=bool(row["breakout"]),
        volatility_ok=bool(row["volatility_ok"]),
        score=int(row["score"]),
        max_score=int(row["max_score"]),
    )


class PredictorStrategy(Strategy):
    """Confluence strategy combining trend, momentum, RSI, breakout and volatility.

    Long-only. Goes fully long when the per-bar signal score reaches
    ``min_score`` (default 4 of 5) and the strategy is flat. Exits on any of:

    * close < ``fast_trend`` EMA (trend break);
    * MACD line crosses below its signal line;
    * close drops below ``entry_price - atr_stop_mult * ATR_at_entry``
      (a volatility-scaled hard stop).
    """

    name = "predictor"

    def __init__(
        self,
        *,
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
        atr_stop_mult: float = 2.0,
        min_score: int = 4,
    ) -> None:
        if fast_trend <= 0 or slow_trend <= 0:
            raise ValueError("fast_trend and slow_trend must be positive")
        if fast_trend >= slow_trend:
            raise ValueError("fast_trend must be strictly smaller than slow_trend")
        if not 0 < rsi_buy_min < rsi_buy_max < 100:
            raise ValueError("expected 0 < rsi_buy_min < rsi_buy_max < 100")
        if not 0 < atr_pct_min < atr_pct_max:
            raise ValueError("expected 0 < atr_pct_min < atr_pct_max")
        if atr_stop_mult <= 0:
            raise ValueError("atr_stop_mult must be positive")
        if not 1 <= min_score <= 5:
            raise ValueError("min_score must be between 1 and 5 (inclusive)")

        self.fast_trend = int(fast_trend)
        self.slow_trend = int(slow_trend)
        self.macd_fast = int(macd_fast)
        self.macd_slow = int(macd_slow)
        self.macd_signal = int(macd_signal)
        self.rsi_period = int(rsi_period)
        self.rsi_buy_min = float(rsi_buy_min)
        self.rsi_buy_max = float(rsi_buy_max)
        self.donchian_period = int(donchian_period)
        self.atr_period = int(atr_period)
        self.atr_pct_min = float(atr_pct_min)
        self.atr_pct_max = float(atr_pct_max)
        self.atr_stop_mult = float(atr_stop_mult)
        self.min_score = int(min_score)

        # Streaming buffer of recent OHLC bars; we recompute indicators on it
        # whenever a new bar arrives. Keeping a small buffer (a few hundred
        # bars) is enough for indicators with periods up to ``slow_trend``.
        self._lookback = max(
            self.slow_trend,
            self.macd_slow + self.macd_signal,
            self.donchian_period + 1,
            self.atr_period + 1,
        ) + 5
        self._buf_open: deque[float] = deque(maxlen=self._lookback)
        self._buf_high: deque[float] = deque(maxlen=self._lookback)
        self._buf_low: deque[float] = deque(maxlen=self._lookback)
        self._buf_close: deque[float] = deque(maxlen=self._lookback)
        self._prev_macd_diff: float | None = None
        self._entry_price: float | None = None
        self._entry_atr: float | None = None

    def reset(self) -> None:
        self._buf_open.clear()
        self._buf_high.clear()
        self._buf_low.clear()
        self._buf_close.clear()
        self._prev_macd_diff = None
        self._entry_price = None
        self._entry_atr = None

    def describe(self) -> dict:
        return {
            "name": self.name,
            "fast_trend": self.fast_trend,
            "slow_trend": self.slow_trend,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "rsi_period": self.rsi_period,
            "rsi_buy_min": self.rsi_buy_min,
            "rsi_buy_max": self.rsi_buy_max,
            "donchian_period": self.donchian_period,
            "atr_period": self.atr_period,
            "atr_pct_min": self.atr_pct_min,
            "atr_pct_max": self.atr_pct_max,
            "atr_stop_mult": self.atr_stop_mult,
            "min_score": self.min_score,
        }

    def _buffered_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": list(self._buf_open),
                "high": list(self._buf_high),
                "low": list(self._buf_low),
                "close": list(self._buf_close),
            }
        )

    def on_bar(self, bar: pd.Series, ctx: StrategyContext) -> list[Order]:
        self._buf_open.append(float(bar["open"]))
        self._buf_high.append(float(bar["high"]))
        self._buf_low.append(float(bar["low"]))
        self._buf_close.append(float(bar["close"]))

        if len(self._buf_close) < self._lookback:
            return []

        table = evaluate_signals(
            self._buffered_frame(),
            fast_trend=self.fast_trend,
            slow_trend=self.slow_trend,
            macd_fast=self.macd_fast,
            macd_slow=self.macd_slow,
            macd_signal=self.macd_signal,
            rsi_period=self.rsi_period,
            rsi_buy_min=self.rsi_buy_min,
            rsi_buy_max=self.rsi_buy_max,
            donchian_period=self.donchian_period,
            atr_period=self.atr_period,
            atr_pct_min=self.atr_pct_min,
            atr_pct_max=self.atr_pct_max,
        )
        last = table.iloc[-1]
        if last[["fast_ema", "slow_ema", "macd", "macd_signal", "atr"]].isna().any():
            return []

        macd_diff = float(last["macd"] - last["macd_signal"])
        close = float(last["close"])
        atr_value = float(last["atr"])
        score = int(last["score"])

        orders: list[Order] = []

        if ctx.position == 0 and ctx.cash > 0:
            if score >= self.min_score:
                orders.append(Order(side=OrderSide.BUY, cash_fraction=1.0))
                # Record the entry context for the stop-loss; the engine fills
                # at the next bar's open, so this is an approximation that's
                # well within typical slippage anyway.
                self._entry_price = close
                self._entry_atr = atr_value
        elif ctx.position > 0:
            stop_hit = (
                self._entry_price is not None
                and self._entry_atr is not None
                and close < self._entry_price - self.atr_stop_mult * self._entry_atr
            )
            trend_break = close < float(last["fast_ema"])
            macd_bear_cross = (
                self._prev_macd_diff is not None
                and self._prev_macd_diff >= 0 > macd_diff
            )
            if stop_hit or trend_break or macd_bear_cross:
                orders.append(Order(side=OrderSide.SELL, position_fraction=1.0))
                self._entry_price = None
                self._entry_atr = None

        self._prev_macd_diff = macd_diff
        return orders


def latest_signal_score(bars: pd.DataFrame, **kwargs: object) -> SignalScore:
    """Return the :class:`SignalScore` for the most recent bar in ``bars``.

    Convenience wrapper around :func:`evaluate_signals`. Raises
    :class:`ValueError` if ``bars`` is too short for the configured indicator
    windows (any indicator still in its warmup period yields NaN).
    """
    table = evaluate_signals(bars, **kwargs)  # type: ignore[arg-type]
    last = table.iloc[-1]
    if last[["fast_ema", "slow_ema", "macd", "macd_signal", "atr"]].isna().any():
        raise ValueError(
            "not enough bars for the configured indicators (need at least "
            f"{max(int(table['max_score'].iloc[-1]), 200)} bars)"
        )
    return _last_score(last)
