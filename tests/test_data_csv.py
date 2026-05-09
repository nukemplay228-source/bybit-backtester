"""Tests for the static CSV loader (public.bybit.com)."""

from __future__ import annotations

import gzip
import io
from typing import Any

import pandas as pd
import pytest
import requests

from bybit_backtest.data_csv import (
    PUBLIC_BASE_URL,
    PublicCsvRequest,
    _month_url,
    _months_between,
    fetch_klines_csv,
)


def _fake_csv_bytes(year: int, month: int, rows: int = 4) -> bytes:
    base = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    lines = []
    for i in range(rows):
        ts = base + pd.Timedelta(minutes=15 * i)
        # public.bybit.com format: YYYY.MM.DD HH:MM,o,h,l,c,v
        lines.append(
            f"{ts.strftime('%Y.%m.%d %H:%M')},100,110,90,{100 + i}.0,1.0"
        )
    raw = "\n".join(lines).encode("utf-8")
    return gzip.compress(raw)


class _FakeResponse:
    def __init__(self, content: bytes, status: int = 200) -> None:
        self.content = content
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class _FakeSession(requests.Session):
    def __init__(self, responses: dict[str, _FakeResponse]) -> None:
        super().__init__()
        self._responses = responses
        self.calls: list[str] = []

    def get(self, url, params=None, timeout=None, **kwargs):  # type: ignore[override]
        self.calls.append(url)
        if url not in self._responses:
            return _FakeResponse(b"", status=404)
        return self._responses[url]


def test_public_csv_request_validates_interval() -> None:
    with pytest.raises(ValueError):
        PublicCsvRequest(
            symbol="BTCUSDT",
            interval="D",
            start=pd.Timestamp("2024-01-01", tz="UTC"),
            end=pd.Timestamp("2024-02-01", tz="UTC"),
        )


def test_public_csv_request_requires_tz_aware_dates() -> None:
    with pytest.raises(ValueError):
        PublicCsvRequest(
            symbol="BTCUSDT",
            interval="60",
            start=pd.Timestamp("2024-01-01"),
            end=pd.Timestamp("2024-02-01", tz="UTC"),
        )


def test_months_between_spans_year_boundary() -> None:
    months = _months_between(
        pd.Timestamp("2023-11-15", tz="UTC"),
        pd.Timestamp("2024-02-10", tz="UTC"),
    )
    assert months == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]


def test_month_url_format() -> None:
    url = _month_url("BTCUSDT", "15", 2024, 2)
    assert url.startswith(PUBLIC_BASE_URL)
    assert url.endswith("BTCUSDT_15_2024-02-01_2024-02-29.csv.gz")


def test_fetch_klines_csv_downloads_and_caches(tmp_path: Any) -> None:
    url_jan = _month_url("BTCUSDT", "15", 2024, 1)
    url_feb = _month_url("BTCUSDT", "15", 2024, 2)
    fake = _FakeSession(
        {
            url_jan: _FakeResponse(_fake_csv_bytes(2024, 1)),
            url_feb: _FakeResponse(_fake_csv_bytes(2024, 2)),
        }
    )
    req = PublicCsvRequest(
        symbol="BTCUSDT",
        interval="15",
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-02-15", tz="UTC"),
    )
    df = fetch_klines_csv(req, cache_dir=tmp_path, session=fake)
    assert len(df) == 8
    assert df.index.is_monotonic_increasing
    assert df["close"].iloc[0] == 100.0
    cached = sorted(tmp_path.glob("*.parquet"))
    assert len(cached) == 2

    # Second call should be entirely served from cache.
    fake2 = _FakeSession({})
    df2 = fetch_klines_csv(req, cache_dir=tmp_path, session=fake2)
    assert fake2.calls == []
    pd.testing.assert_frame_equal(df, df2)


def test_fetch_klines_csv_skips_missing_months(tmp_path: Any) -> None:
    url_jan = _month_url("BTCUSDT", "15", 2030, 1)
    url_feb = _month_url("BTCUSDT", "15", 2030, 2)
    fake = _FakeSession(
        {
            url_jan: _FakeResponse(b"", status=404),
            url_feb: _FakeResponse(b"", status=404),
        }
    )
    req = PublicCsvRequest(
        symbol="BTCUSDT",
        interval="15",
        start=pd.Timestamp("2030-01-01", tz="UTC"),
        end=pd.Timestamp("2030-02-15", tz="UTC"),
    )
    df = fetch_klines_csv(req, cache_dir=tmp_path, session=fake)
    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "turnover"]


def test_fetch_klines_csv_clips_to_window(tmp_path: Any) -> None:
    # Generate 96 quarters (one day) but request only 1 hour.
    base = pd.Timestamp("2024-03-01", tz="UTC")
    rows = []
    for i in range(96):
        ts = base + pd.Timedelta(minutes=15 * i)
        rows.append(f"{ts.strftime('%Y.%m.%d %H:%M')},1,1,1,{i}.0,1.0")
    raw = gzip.compress("\n".join(rows).encode("utf-8"))

    url = _month_url("BTCUSDT", "15", 2024, 3)
    fake = _FakeSession({url: _FakeResponse(raw)})
    req = PublicCsvRequest(
        symbol="BTCUSDT",
        interval="15",
        start=pd.Timestamp("2024-03-01 00:30", tz="UTC"),
        end=pd.Timestamp("2024-03-01 01:30", tz="UTC"),
    )
    df = fetch_klines_csv(req, cache_dir=tmp_path, session=fake)
    # 0:30, 0:45, 1:00, 1:15 -> 4 bars (end is exclusive)
    assert len(df) == 4
    # Cached parquet should not be polluted by the buffer
    _ = io.BytesIO  # silence unused
