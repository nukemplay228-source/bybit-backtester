"""Tests for the data loader (no live network calls)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest
import requests

from bybit_backtest.data import (
    BYBIT_BASE_URL,
    INTERVAL_MS,
    KLINE_ENDPOINT,
    KlineRequest,
    fetch_klines,
    parse_date,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession(requests.Session):
    """Session subclass that intercepts ``get`` and returns canned bars."""

    def __init__(self, batches: list[list[list[str]]]) -> None:
        super().__init__()
        self._batches = list(batches)
        self.calls: list[dict[str, Any]] = []

    def get(self, url, params=None, timeout=None, **kwargs):  # type: ignore[override]
        assert url == BYBIT_BASE_URL + KLINE_ENDPOINT
        self.calls.append(dict(params or {}))
        if not self._batches:
            return _FakeResponse({"retCode": 0, "result": {"list": []}})
        return _FakeResponse({"retCode": 0, "result": {"list": self._batches.pop(0)}})


def _kline_row(ts_ms: int, close: float) -> list[str]:
    return [str(ts_ms), "100", "110", "90", str(close), "1", "100"]


def test_parse_date_handles_iso_dates() -> None:
    assert parse_date("2024-01-01") == 1704067200000


def test_kline_request_validation() -> None:
    with pytest.raises(ValueError):
        KlineRequest(symbol="BTCUSDT", interval="bogus", start_ms=0, end_ms=1)
    with pytest.raises(ValueError):
        KlineRequest(symbol="BTCUSDT", interval="60", start_ms=10, end_ms=10)
    with pytest.raises(ValueError):
        KlineRequest(symbol="BTCUSDT", interval="60", start_ms=0, end_ms=1, category="bogus")


def test_fetch_klines_uses_session_and_paginates(tmp_path) -> None:
    interval = "60"
    bar_ms = INTERVAL_MS[interval]
    start_ms = 1_700_000_000_000
    # Two pages of 1000 + a partial third page.
    batch_1 = [_kline_row(start_ms + i * bar_ms, 100 + i) for i in range(1000)]
    batch_1.reverse()  # API returns newest-first
    batch_2 = [_kline_row(start_ms + (1000 + i) * bar_ms, 200 + i) for i in range(1000)]
    batch_2.reverse()
    batch_3 = [_kline_row(start_ms + (2000 + i) * bar_ms, 300 + i) for i in range(50)]
    batch_3.reverse()

    end_ms = start_ms + 2050 * bar_ms

    fake = _FakeSession([batch_1, batch_2, batch_3])
    req = KlineRequest(
        symbol="BTCUSDT",
        interval=interval,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    df = fetch_klines(
        req,
        cache_dir=tmp_path,
        use_cache=False,
        sleep_between_calls=0.0,
        session=fake,
    )
    assert len(df) == 2050
    assert df.index.is_monotonic_increasing
    assert df["close"].iloc[0] == 100.0
    assert df["close"].iloc[-1] == 349.0
    # 3 paginated calls expected.
    assert len(fake.calls) == 3


def test_fetch_klines_writes_cache(tmp_path) -> None:
    interval = "D"
    bar_ms = INTERVAL_MS[interval]
    start_ms = 1_700_000_000_000
    batch = [_kline_row(start_ms + i * bar_ms, 100 + i) for i in range(5)]
    batch.reverse()
    fake = _FakeSession([batch])
    req = KlineRequest(
        symbol="ETHUSDT",
        interval=interval,
        start_ms=start_ms,
        end_ms=start_ms + 5 * bar_ms,
    )
    df1 = fetch_klines(req, cache_dir=tmp_path, sleep_between_calls=0.0, session=fake)
    cached = list(tmp_path.glob("*.parquet"))
    assert len(cached) == 1

    # Second call should not hit the network: empty session must still work.
    fake_empty = _FakeSession([])
    df2 = fetch_klines(req, cache_dir=tmp_path, sleep_between_calls=0.0, session=fake_empty)
    assert fake_empty.calls == []
    pd.testing.assert_frame_equal(df1, df2)


def test_fetch_klines_raises_on_api_error(tmp_path) -> None:
    class ErrSession(requests.Session):
        def get(self, *args, **kwargs):  # type: ignore[override]
            return _FakeResponse(
                {"retCode": 10001, "retMsg": "boom", "result": {"list": []}}
            )

    req = KlineRequest(symbol="BTCUSDT", interval="60", start_ms=0, end_ms=3_600_000)
    with pytest.raises(RuntimeError):
        fetch_klines(
            req,
            cache_dir=tmp_path,
            use_cache=False,
            sleep_between_calls=0.0,
            session=ErrSession(),
        )


def test_kline_request_payload_serializable() -> None:
    req = KlineRequest(symbol="BTCUSDT", interval="60", start_ms=1, end_ms=2)
    # Make sure the dataclass can be turned into JSON for logging.
    json.dumps(
        {
            "symbol": req.symbol,
            "interval": req.interval,
            "start_ms": req.start_ms,
            "end_ms": req.end_ms,
            "category": req.category,
        }
    )
