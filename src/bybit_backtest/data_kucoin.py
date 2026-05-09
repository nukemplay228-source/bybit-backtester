"""KuCoin spot, perpetual, and funding-rate loaders.

Bybit's V5 API and Binance's perpetuals API are both blocked from the cloud
egress IP we run on, and from many residential IPs in geo-restricted regions.
KuCoin's public API is one of the few that:

* serves USDT-margined perpetual klines and funding-rate history without auth,
* keeps history all the way back to 2021 for the major pairs,
* is reachable from the same locations where Bybit is blocked.

Funding rates between top venues track each other very closely (because
arbitrageurs literally close the gap), so KuCoin XBTUSDTM funding is a
reasonable proxy for Bybit BTCUSDT funding for a "is this strategy class
viable at all" backtest. The README documents this caveat explicitly.

All three loaders cache their output to parquet under ``~/.cache/bybit_backtest``
so subsequent runs are instant.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

KUCOIN_SPOT_BASE_URL = "https://api.kucoin.com"
KUCOIN_FUTURES_BASE_URL = "https://api-futures.kucoin.com"

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "bybit_backtest"

SPOT_INTERVALS: dict[str, tuple[str, int]] = {
    "1": ("1min", 60_000),
    "5": ("5min", 5 * 60_000),
    "15": ("15min", 15 * 60_000),
    "30": ("30min", 30 * 60_000),
    "60": ("1hour", 60 * 60_000),
    "240": ("4hour", 4 * 60 * 60_000),
    "480": ("8hour", 8 * 60 * 60_000),
    "D": ("1day", 24 * 60 * 60_000),
}

PERP_INTERVAL_MIN: dict[str, int] = {
    "1": 1,
    "5": 5,
    "15": 15,
    "30": 30,
    "60": 60,
    "240": 240,
    "480": 480,
    "D": 1440,
}


def _cache_path(name: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / name


def _read_cached(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def _empty_kline_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return pd.DataFrame(
        {col: pd.Series(dtype="float64") for col in ("open", "high", "low", "close", "volume")},
        index=idx,
    )


def _empty_funding_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return pd.DataFrame({"funding_rate": pd.Series(dtype="float64")}, index=idx)


def _to_ms(value: str) -> int:
    ts = (
        pd.Timestamp(value, tz="UTC")
        if "T" in value or " " in value
        else pd.Timestamp(value + " 00:00:00", tz="UTC")
    )
    return int(ts.value // 1_000_000)


def fetch_spot_klines(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    sleep_between_calls: float = 0.15,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download (and cache) KuCoin spot klines.

    ``symbol`` should be a KuCoin spot symbol such as ``BTC-USDT``.
    """
    if interval not in SPOT_INTERVALS:
        raise ValueError(
            f"interval must be one of {sorted(SPOT_INTERVALS)}, got {interval!r}"
        )

    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    start_ms = _to_ms(start)
    end_ms = _to_ms(end)
    if start_ms >= end_ms:
        raise ValueError("start must be strictly less than end")
    cache_file = _cache_path(
        f"kucoin_spot_{symbol}_{interval}_{start_ms}_{end_ms}.parquet", cache_dir
    )
    if use_cache and cache_file.exists():
        logger.info("Using cached KuCoin spot klines: %s", cache_file)
        return _read_cached(cache_file)

    own_session = session is None
    session = session or requests.Session()

    kucoin_type, bar_ms = SPOT_INTERVALS[interval]
    chunk_ms = bar_ms * 1500

    rows: list[list[str]] = []
    chunk_end = end_ms
    while chunk_end > start_ms:
        chunk_start = max(start_ms, chunk_end - chunk_ms)
        params = {
            "symbol": symbol,
            "type": kucoin_type,
            "startAt": chunk_start // 1000,
            "endAt": chunk_end // 1000,
        }
        response = session.get(
            KUCOIN_SPOT_BASE_URL + "/api/v1/market/candles",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "200000":
            raise RuntimeError(f"KuCoin spot kline error: {payload}")
        batch = payload.get("data") or []
        if not batch:
            break
        rows.extend(batch)
        oldest_in_batch = int(batch[-1][0]) * 1000
        if oldest_in_batch <= start_ms:
            break
        chunk_end = oldest_in_batch
        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    if own_session:
        session.close()

    df = _spot_to_dataframe(rows, start_ms=start_ms, end_ms=end_ms)
    if use_cache and not df.empty:
        df.to_parquet(cache_file)
    return df


def _spot_to_dataframe(rows: list[list[str]], *, start_ms: int, end_ms: int) -> pd.DataFrame:
    if not rows:
        return _empty_kline_frame()
    df = pd.DataFrame(
        rows,
        columns=["time", "open", "close", "high", "low", "volume", "turnover"],
    )
    df["start_ms"] = pd.to_numeric(df["time"], errors="raise").astype("int64") * 1000
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype("float64")
    df = df.drop_duplicates(subset=["start_ms"]).sort_values("start_ms").reset_index(drop=True)
    mask = (df["start_ms"] >= start_ms) & (df["start_ms"] < end_ms)
    df = df.loc[mask].copy()
    df["timestamp"] = pd.to_datetime(df["start_ms"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    return df[["open", "high", "low", "close", "volume"]]


def fetch_perp_klines(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    sleep_between_calls: float = 0.15,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download (and cache) KuCoin USDT-margined perpetual klines.

    ``symbol`` should be a KuCoin futures symbol such as ``XBTUSDTM``.
    """
    if interval not in PERP_INTERVAL_MIN:
        raise ValueError(
            f"interval must be one of {sorted(PERP_INTERVAL_MIN)}, got {interval!r}"
        )

    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    start_ms = _to_ms(start)
    end_ms = _to_ms(end)
    if start_ms >= end_ms:
        raise ValueError("start must be strictly less than end")
    cache_file = _cache_path(
        f"kucoin_perp_{symbol}_{interval}_{start_ms}_{end_ms}.parquet", cache_dir
    )
    if use_cache and cache_file.exists():
        logger.info("Using cached KuCoin perp klines: %s", cache_file)
        return _read_cached(cache_file)

    own_session = session is None
    session = session or requests.Session()

    granularity = PERP_INTERVAL_MIN[interval]
    bar_ms = granularity * 60_000
    chunk_ms = bar_ms * 200

    rows: list[list[float]] = []
    chunk_start = start_ms
    while chunk_start < end_ms:
        chunk_end = min(end_ms, chunk_start + chunk_ms)
        params = {
            "symbol": symbol,
            "granularity": granularity,
            "from": chunk_start,
            "to": chunk_end,
        }
        response = session.get(
            KUCOIN_FUTURES_BASE_URL + "/api/v1/kline/query",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "200000":
            raise RuntimeError(f"KuCoin perp kline error: {payload}")
        batch = payload.get("data") or []
        if not batch:
            break
        rows.extend(batch)
        newest_in_batch = int(batch[-1][0])
        next_start = newest_in_batch + bar_ms
        if next_start <= chunk_start:
            next_start = chunk_start + chunk_ms
        chunk_start = next_start
        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    if own_session:
        session.close()

    df = _perp_to_dataframe(rows, start_ms=start_ms, end_ms=end_ms)
    if use_cache and not df.empty:
        df.to_parquet(cache_file)
    return df


def _perp_to_dataframe(
    rows: list[list[float]], *, start_ms: int, end_ms: int
) -> pd.DataFrame:
    if not rows:
        return _empty_kline_frame()
    df = pd.DataFrame(
        rows,
        columns=["start_ms", "open", "high", "low", "close", "volume", "turnover"],
    )
    df["start_ms"] = pd.to_numeric(df["start_ms"], errors="raise").astype("int64")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype("float64")
    df = df.drop_duplicates(subset=["start_ms"]).sort_values("start_ms").reset_index(drop=True)
    mask = (df["start_ms"] >= start_ms) & (df["start_ms"] < end_ms)
    df = df.loc[mask].copy()
    df["timestamp"] = pd.to_datetime(df["start_ms"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    return df[["open", "high", "low", "close", "volume"]]


def fetch_funding_rates(
    symbol: str,
    start: str,
    end: str,
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    sleep_between_calls: float = 0.15,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download (and cache) KuCoin funding-rate history.

    Returns a DataFrame indexed by funding-payment timestamp (UTC) with a
    single ``funding_rate`` column. KuCoin's funding interval is 8 hours,
    matching Bybit's settlement schedule.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    start_ms = _to_ms(start)
    end_ms = _to_ms(end)
    if start_ms >= end_ms:
        raise ValueError("start must be strictly less than end")
    cache_file = _cache_path(
        f"kucoin_funding_{symbol}_{start_ms}_{end_ms}.parquet", cache_dir
    )
    if use_cache and cache_file.exists():
        logger.info("Using cached KuCoin funding rates: %s", cache_file)
        return _read_cached(cache_file)

    own_session = session is None
    session = session or requests.Session()

    chunk_ms = 30 * 24 * 60 * 60_000
    rows: list[dict[str, float | int]] = []
    chunk_start = start_ms
    while chunk_start < end_ms:
        chunk_end = min(end_ms, chunk_start + chunk_ms)
        params = {
            "symbol": symbol,
            "from": chunk_start,
            "to": chunk_end,
        }
        response = session.get(
            KUCOIN_FUTURES_BASE_URL + "/api/v1/contract/funding-rates",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "200000":
            raise RuntimeError(f"KuCoin funding-rate error: {payload}")
        batch = payload.get("data") or []
        rows.extend(batch)
        chunk_start = chunk_end
        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    if own_session:
        session.close()

    df = _funding_to_dataframe(rows, start_ms=start_ms, end_ms=end_ms)
    if use_cache and not df.empty:
        df.to_parquet(cache_file)
    return df


def _funding_to_dataframe(
    rows: list[dict[str, float | int]], *, start_ms: int, end_ms: int
) -> pd.DataFrame:
    if not rows:
        return _empty_funding_frame()
    df = pd.DataFrame(rows)
    df["timepoint"] = pd.to_numeric(df["timepoint"], errors="raise").astype("int64")
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="raise").astype("float64")
    df = df.drop_duplicates(subset=["timepoint"]).sort_values("timepoint").reset_index(drop=True)
    mask = (df["timepoint"] >= start_ms) & (df["timepoint"] < end_ms)
    df = df.loc[mask].copy()
    df["timestamp"] = pd.to_datetime(df["timepoint"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    return df[["funding_rate"]]
