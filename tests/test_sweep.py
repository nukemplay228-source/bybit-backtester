"""Tests for the parameter-sweep and walk-forward helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bybit_backtest.sweep import (
    SUPPORTED_METRICS,
    _expand_grid,
    _split_windows,
    parameter_sweep,
    walk_forward,
)


def _make_trending_bars(periods: int) -> pd.DataFrame:
    rng = pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")
    closes = np.linspace(100.0, 200.0, periods)
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


def test_expand_grid_returns_cartesian_product() -> None:
    grid = {"fast": [10, 20], "slow": [50, 100, 200]}
    combos = _expand_grid(grid)
    assert len(combos) == 6
    pairs = {(c["fast"], c["slow"]) for c in combos}
    assert pairs == {
        (10, 50), (10, 100), (10, 200),
        (20, 50), (20, 100), (20, 200),
    }


def test_expand_grid_empty_returns_one_empty_combo() -> None:
    assert _expand_grid({}) == [{}]


def test_expand_grid_single_value_wrapped() -> None:
    combos = _expand_grid({"fast": 10, "slow": [50, 100]})
    assert combos == [{"fast": 10, "slow": 50}, {"fast": 10, "slow": 100}]


def test_split_windows_creates_disjoint_test_windows() -> None:
    bars = _make_trending_bars(100)
    chunks = _split_windows(bars, train_bars=40, test_bars=20)
    assert len(chunks) == 3
    for train, test in chunks:
        assert len(train) == 40
        assert len(test) == 20
        assert train.index[-1] < test.index[0]
    # Subsequent test windows are non-overlapping.
    test_starts = [c[1].index[0] for c in chunks]
    assert test_starts == sorted(test_starts)


def test_split_windows_raises_when_data_too_short() -> None:
    bars = _make_trending_bars(50)
    with pytest.raises(ValueError):
        _split_windows(bars, train_bars=40, test_bars=20)


def test_parameter_sweep_runs_each_combination() -> None:
    bars = _make_trending_bars(120)
    results = parameter_sweep(
        bars,
        "sma_cross",
        {"fast": [5, 10], "slow": [20, 30]},
    )
    assert len(results) == 4
    assert {"fast", "slow", "total_return_pct", "sharpe", "max_drawdown_pct"}.issubset(
        results.columns
    )


def test_parameter_sweep_skips_invalid_combos() -> None:
    bars = _make_trending_bars(120)
    results = parameter_sweep(
        bars,
        "sma_cross",
        {"fast": [10, 20, 50], "slow": [20, 30]},  # fast>=slow combos invalid
    )
    valid_pairs = {(int(r["fast"]), int(r["slow"])) for _, r in results.iterrows()}
    for f, s in valid_pairs:
        assert f < s


def test_parameter_sweep_empty_grid_returns_single_row() -> None:
    bars = _make_trending_bars(120)
    results = parameter_sweep(bars, "sma_cross", {})
    assert len(results) == 1


def test_walk_forward_produces_train_and_test_columns() -> None:
    bars = _make_trending_bars(200)
    wf = walk_forward(
        bars,
        "sma_cross",
        {"fast": [5, 10], "slow": [20, 30]},
        train_bars=80,
        test_bars=40,
        metric="total_return_pct",
    )
    assert not wf.empty
    expected = {
        "window_idx",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "best_params",
        "train_total_return_pct",
        "test_total_return_pct",
        "test_max_drawdown_pct",
        "test_sharpe",
        "test_num_trades",
    }
    assert expected.issubset(wf.columns)
    # Test windows must be strictly after their training window.
    for _, row in wf.iterrows():
        assert row["train_end"] < row["test_start"]


def test_walk_forward_rejects_unknown_metric() -> None:
    bars = _make_trending_bars(200)
    with pytest.raises(ValueError):
        walk_forward(
            bars,
            "sma_cross",
            {"fast": [5], "slow": [20]},
            train_bars=80,
            test_bars=40,
            metric="not_a_metric",
        )


def test_supported_metrics_are_attributes_of_report() -> None:
    bars = _make_trending_bars(200)
    wf = walk_forward(
        bars,
        "sma_cross",
        {"fast": [5], "slow": [20]},
        train_bars=80,
        test_bars=40,
        metric="sharpe",
    )
    # Every metric in SUPPORTED_METRICS must round-trip without error.
    for metric in SUPPORTED_METRICS:
        wf2 = walk_forward(
            bars,
            "sma_cross",
            {"fast": [5], "slow": [20]},
            train_bars=80,
            test_bars=40,
            metric=metric,
        )
        assert f"test_{metric}" in wf2.columns
    assert "test_sharpe" in wf.columns
