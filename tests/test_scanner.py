"""Tests for the multi-asset scanner."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bybit_backtest.scanner import scan_assets


def _make_bars(periods: int, *, trend: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")
    rs = np.random.RandomState(seed)
    noise = rs.randn(periods) * 0.5
    drift = np.linspace(0.0, trend, periods)
    closes = 100.0 + drift + np.cumsum(noise) * 0.1
    closes = np.maximum(closes, 1.0)
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.002,
            "low": closes * 0.998,
            "close": closes,
            "volume": np.full(periods, 1.0),
            "turnover": np.full(periods, 100.0),
        },
        index=rng,
    )
    df.index.name = "timestamp"
    return df


def _stub_fetcher(symbol_to_bars: dict[str, pd.DataFrame]):
    def _fetch(symbol: str, interval: str, start: str, end: str, *, use_cache: bool = True):
        return symbol_to_bars.get(symbol, pd.DataFrame())

    return _fetch


def test_scan_returns_one_row_per_symbol_sorted_by_sharpe() -> None:
    bars_a = _make_bars(150, trend=50.0, seed=1)
    bars_b = _make_bars(150, trend=-50.0, seed=2)
    bars_c = _make_bars(150, trend=20.0, seed=3)
    fetcher = _stub_fetcher({"AAA": bars_a, "BBB": bars_b, "CCC": bars_c})
    result = scan_assets(
        ["AAA", "BBB", "CCC"],
        "60",
        "2024-01-01",
        "2024-12-01",
        "sma_cross",
        {"fast": 5, "slow": 20},
        fetcher=fetcher,
    )
    assert list(result["symbol"]) == sorted(
        result["symbol"], key=lambda s: -result.set_index("symbol").loc[s, "sharpe"]
    )
    assert (result["error"] == "").all()
    assert (result["num_bars"] == 150).all()


def test_scan_records_no_data_error_for_missing_symbols() -> None:
    bars_a = _make_bars(150, trend=20.0, seed=1)
    fetcher = _stub_fetcher({"AAA": bars_a})  # ZZZ deliberately missing
    result = scan_assets(
        ["AAA", "ZZZ"],
        "60",
        "2024-01-01",
        "2024-12-01",
        "sma_cross",
        {"fast": 5, "slow": 20},
        fetcher=fetcher,
    )
    zzz_row = result[result["symbol"] == "ZZZ"].iloc[0]
    assert zzz_row["error"] == "no_data"
    assert pd.isna(zzz_row["total_return_pct"])
    aaa_row = result[result["symbol"] == "AAA"].iloc[0]
    assert aaa_row["error"] == ""


def test_scan_records_fetch_exceptions_without_crashing() -> None:
    def _broken_fetcher(symbol, interval, start, end, *, use_cache=True):
        if symbol == "BAD":
            raise RuntimeError("boom")
        return _make_bars(150, trend=20.0, seed=4)

    result = scan_assets(
        ["GOOD", "BAD"],
        "60",
        "2024-01-01",
        "2024-12-01",
        "sma_cross",
        {"fast": 5, "slow": 20},
        fetcher=_broken_fetcher,
    )
    bad_row = result[result["symbol"] == "BAD"].iloc[0]
    assert "boom" in bad_row["error"]
    assert pd.isna(bad_row["total_return_pct"])
    good_row = result[result["symbol"] == "GOOD"].iloc[0]
    assert good_row["error"] == ""
