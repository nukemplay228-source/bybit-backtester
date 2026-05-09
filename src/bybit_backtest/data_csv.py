"""Alternative kline loader that pulls gzipped CSVs from public.bybit.com.

Bybit's V5 API (``api.bybit.com``) is geo-blocked from many cloud regions
(returns HTTP 403). The static CSV archives served from
``https://public.bybit.com/kline_for_metatrader4/<SYMBOL>/<YEAR>/`` are
hosted on a different CDN that does not enforce the same geo-restrictions
and so make a useful fallback.

Each archive contains a single month of klines for a fixed interval, with
the format::

    2024.01.01 00:00,open,high,low,close,volume

Only the ``1, 5, 15, 30, 60`` minute intervals are published (per Bybit's
MT4 export). For daily/weekly/monthly bars use :mod:`bybit_backtest.data`.
"""

from __future__ import annotations

import gzip
import io
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

PUBLIC_BASE_URL = "https://public.bybit.com/kline_for_metatrader4"
SUPPORTED_INTERVALS = {"1", "5", "15", "30", "60"}

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "bybit_backtest" / "public_csv"


@dataclass(frozen=True)
class PublicCsvRequest:
    """Request parameters for the static CSV archive."""

    symbol: str
    interval: str
    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        if self.interval not in SUPPORTED_INTERVALS:
            raise ValueError(
                f"interval must be one of {sorted(SUPPORTED_INTERVALS)}, "
                f"got {self.interval!r}. Use the API loader for daily+ intervals."
            )
        if self.start.tz is None or self.end.tz is None:
            raise ValueError("start and end must be timezone-aware (UTC)")
        if self.start >= self.end:
            raise ValueError("start must be strictly less than end")


def _month_url(symbol: str, interval: str, year: int, month: int) -> str:
    last_day = pd.Period(f"{year}-{month:02d}", freq="M").end_time.day
    fname = (
        f"{symbol}_{interval}_{year:04d}-{month:02d}-01"
        f"_{year:04d}-{month:02d}-{last_day:02d}.csv.gz"
    )
    return f"{PUBLIC_BASE_URL}/{symbol}/{year}/{fname}"


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    current = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    end_excl = end
    while current < end_excl:
        months.append((current.year, current.month))
        if current.month == 12:
            current = pd.Timestamp(year=current.year + 1, month=1, day=1, tz="UTC")
        else:
            current = pd.Timestamp(
                year=current.year, month=current.month + 1, day=1, tz="UTC"
            )
    return months


def _parse_csv_bytes(raw: bytes) -> pd.DataFrame:
    text = gzip.decompress(raw).decode("utf-8")
    df = pd.read_csv(
        io.StringIO(text),
        header=None,
        names=["timestamp", "open", "high", "low", "close", "volume"],
        dtype={
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        },
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y.%m.%d %H:%M", utc=True)
    df = df.set_index("timestamp").sort_index()
    df["turnover"] = df["close"] * df["volume"]
    return df[["open", "high", "low", "close", "volume", "turnover"]]


def _fetch_month(
    session: requests.Session,
    symbol: str,
    interval: str,
    year: int,
    month: int,
    *,
    cache_dir: Path,
    use_cache: bool,
    timeout: float,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{symbol}_{interval}_{year:04d}-{month:02d}.parquet"
    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)
    url = _month_url(symbol, interval, year, month)
    logger.info("Fetching %s", url)
    response = session.get(url, timeout=timeout)
    if response.status_code == 404:
        logger.warning("Month not available on public.bybit.com: %s", url)
        return _empty_kline_frame()
    response.raise_for_status()
    df = _parse_csv_bytes(response.content)
    if use_cache and not df.empty:
        df.to_parquet(cache_file)
    return df


def _empty_kline_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            col: pd.Series(dtype="float64")
            for col in ("open", "high", "low", "close", "volume", "turnover")
        },
        index=idx,
    )


def fetch_klines_csv(
    req: PublicCsvRequest,
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    timeout: float = 60.0,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download (and cache) klines from ``public.bybit.com`` for the given range."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    own_session = session is None
    session = session or requests.Session()

    frames: list[pd.DataFrame] = []
    for year, month in _months_between(req.start, req.end):
        frame = _fetch_month(
            session,
            req.symbol,
            req.interval,
            year,
            month,
            cache_dir=cache_dir,
            use_cache=use_cache,
            timeout=timeout,
        )
        if not frame.empty:
            frames.append(frame)

    if own_session:
        session.close()

    if not frames:
        return _empty_kline_frame()
    df = pd.concat(frames).sort_index()
    df = df[(df.index >= req.start) & (df.index < req.end)]
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_klines_csv(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """High-level helper that accepts dates as ISO strings."""
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    req = PublicCsvRequest(symbol=symbol, interval=interval, start=start_ts, end=end_ts)
    return fetch_klines_csv(
        req, cache_dir=cache_dir, use_cache=use_cache, session=session
    )
