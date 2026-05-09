"""Historical kline data loader for Bybit V5 public API.

Fetches OHLCV data from https://api.bybit.com/v5/market/kline and caches
results on disk so subsequent runs are instant.

The endpoint is public and does not require any API key.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BYBIT_BASE_URL = "https://api.bybit.com"
KLINE_ENDPOINT = "/v5/market/kline"

# Mapping of human-friendly intervals to Bybit interval strings and bar duration in ms.
INTERVAL_MS: dict[str, int] = {
    "1": 60_000,
    "3": 3 * 60_000,
    "5": 5 * 60_000,
    "15": 15 * 60_000,
    "30": 30 * 60_000,
    "60": 60 * 60_000,
    "120": 2 * 60 * 60_000,
    "240": 4 * 60 * 60_000,
    "360": 6 * 60 * 60_000,
    "720": 12 * 60 * 60_000,
    "D": 24 * 60 * 60_000,
    "W": 7 * 24 * 60 * 60_000,
    "M": 30 * 24 * 60 * 60_000,
}

VALID_CATEGORIES = {"spot", "linear", "inverse"}

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "bybit_backtest"


@dataclass(frozen=True)
class KlineRequest:
    """Parameters for a kline download request."""

    symbol: str
    interval: str
    start_ms: int
    end_ms: int
    category: str = "spot"

    def __post_init__(self) -> None:
        if self.category not in VALID_CATEGORIES:
            raise ValueError(
                f"category must be one of {sorted(VALID_CATEGORIES)}, got {self.category!r}"
            )
        if self.interval not in INTERVAL_MS:
            raise ValueError(
                f"interval must be one of {sorted(INTERVAL_MS)}, got {self.interval!r}"
            )
        if self.start_ms >= self.end_ms:
            raise ValueError("start_ms must be strictly less than end_ms")


def parse_date(value: str) -> int:
    """Parse a YYYY-MM-DD or full ISO timestamp into UTC millis since epoch."""
    ts = pd.Timestamp(value, tz="UTC") if "T" in value or " " in value else pd.Timestamp(
        value + " 00:00:00", tz="UTC"
    )
    return int(ts.value // 1_000_000)


def _cache_path(req: KlineRequest, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = f"{req.category}_{req.symbol}_{req.interval}_{req.start_ms}_{req.end_ms}.parquet"
    return cache_dir / name


def _request_klines(
    session: requests.Session,
    req: KlineRequest,
    *,
    chunk_start_ms: int,
    limit: int,
    timeout: float,
) -> list[list[str]]:
    """Make a single API call and return the raw kline list."""
    params = {
        "category": req.category,
        "symbol": req.symbol,
        "interval": req.interval,
        "start": chunk_start_ms,
        "end": req.end_ms,
        "limit": limit,
    }
    response = session.get(
        BYBIT_BASE_URL + KLINE_ENDPOINT, params=params, timeout=timeout
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("retCode") != 0:
        raise RuntimeError(
            f"Bybit API error: retCode={payload.get('retCode')} retMsg={payload.get('retMsg')!r}"
        )
    return payload["result"]["list"]


def fetch_klines(
    req: KlineRequest,
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    limit: int = 1000,
    sleep_between_calls: float = 0.15,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download (and cache) klines from Bybit covering the requested range.

    Returns a DataFrame indexed by UTC timestamp with columns
    ``open, high, low, close, volume, turnover``.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_file = _cache_path(req, cache_dir)
    if use_cache and cache_file.exists():
        logger.info("Using cached klines: %s", cache_file)
        return _read_cached(cache_file)

    own_session = session is None
    session = session or requests.Session()

    bar_ms = INTERVAL_MS[req.interval]
    chunk_size_ms = bar_ms * limit

    rows: list[list[str]] = []
    chunk_start = req.start_ms
    while chunk_start < req.end_ms:
        logger.debug(
            "Fetching klines start=%s end=%s symbol=%s interval=%s",
            chunk_start,
            req.end_ms,
            req.symbol,
            req.interval,
        )
        batch = _request_klines(
            session,
            req,
            chunk_start_ms=chunk_start,
            limit=limit,
            timeout=timeout,
        )
        if not batch:
            break
        rows.extend(batch)
        # Bybit returns klines in descending order: oldest is the last item.
        oldest_in_batch = int(batch[-1][0])
        newest_in_batch = int(batch[0][0])
        # Advance the window past the newest bar we got. If we got fewer than
        # `limit` bars, also exit, because there's nothing more to fetch.
        next_start = newest_in_batch + bar_ms
        if next_start <= chunk_start:
            # Should not happen, but guard against an infinite loop.
            next_start = chunk_start + chunk_size_ms
        chunk_start = next_start
        if len(batch) < limit and oldest_in_batch <= req.start_ms:
            break
        if len(batch) < limit:
            # Partial batch usually means no more data ahead.
            break
        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    if own_session:
        session.close()

    df = _to_dataframe(rows, start_ms=req.start_ms, end_ms=req.end_ms)
    if use_cache and not df.empty:
        df.to_parquet(cache_file)
    return df


def _to_dataframe(rows: list[list[str]], *, start_ms: int, end_ms: int) -> pd.DataFrame:
    if not rows:
        return _empty_kline_frame()
    df = pd.DataFrame(
        rows,
        columns=["start", "open", "high", "low", "close", "volume", "turnover"],
    )
    df["start"] = pd.to_numeric(df["start"], errors="raise").astype("int64")
    for col in ("open", "high", "low", "close", "volume", "turnover"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype("float64")
    df = df.drop_duplicates(subset=["start"]).sort_values("start").reset_index(drop=True)
    mask = (df["start"] >= start_ms) & (df["start"] < end_ms)
    df = df.loc[mask].copy()
    df["timestamp"] = pd.to_datetime(df["start"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    return df[["open", "high", "low", "close", "volume", "turnover"]]


def _empty_kline_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return pd.DataFrame(
        {col: pd.Series(dtype="float64") for col in
         ("open", "high", "low", "close", "volume", "turnover")},
        index=idx,
    )


def _read_cached(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def load_klines(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    *,
    category: str = "spot",
    cache_dir: Path | None = None,
    use_cache: bool = True,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """High-level helper that accepts dates as ISO strings."""
    req = KlineRequest(
        symbol=symbol,
        interval=interval,
        start_ms=parse_date(start),
        end_ms=parse_date(end),
        category=category,
    )
    return fetch_klines(req, cache_dir=cache_dir, use_cache=use_cache, session=session)
