"""Walk-forward validation and parameter-grid search.

The point of this module is to fight overfitting. A strategy that looks great
when both its parameters and its performance are measured on the same window
of data is almost guaranteed to disappoint live, because the optimiser found
the parameters that fit *that* window's noise, not a real signal.

Two helpers live here:

* :func:`parameter_sweep` runs the same strategy with every combination of a
  parameter grid against a single window and returns a tidy dataframe of
  results. Useful as a building block and as a quick "did I find anything at
  all?" sanity check.
* :func:`walk_forward` slides a (train, test) window across the data: the grid
  is searched on the train slice, the *single* best combination by a chosen
  metric is then evaluated on the next, untouched test slice. The resulting
  dataframe lets you compare in-sample vs out-of-sample performance window by
  window — a strategy whose train metric is always great but whose test
  metric is around zero (or worse) is overfit.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from bybit_backtest.backtest import run_backtest
from bybit_backtest.strategies import build_strategy

logger = logging.getLogger(__name__)


SUPPORTED_METRICS = ("sharpe", "sortino", "total_return_pct", "cagr_pct")


def _expand_grid(param_grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Return the Cartesian product of ``param_grid`` as a list of param dicts."""
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    values_lists: list[Sequence[Any]] = []
    for k in keys:
        v = param_grid[k]
        if isinstance(v, str) or not isinstance(v, Iterable):
            values_lists.append([v])
        else:
            values_lists.append(list(v))
    combos: list[dict[str, Any]] = []
    for combo in itertools.product(*values_lists):
        combos.append(dict(zip(keys, combo, strict=True)))
    return combos


def _short_params(params: Mapping[str, Any]) -> str:
    if not params:
        return "{}"
    return "{" + ", ".join(f"{k}={params[k]!r}" for k in sorted(params)) + "}"


def parameter_sweep(
    bars: pd.DataFrame,
    strategy_name: str,
    param_grid: Mapping[str, Sequence[Any]],
    *,
    initial_cash: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage: float = 0.0005,
    interval: str = "60",
) -> pd.DataFrame:
    """Run ``strategy_name`` for every combination in ``param_grid`` on ``bars``.

    Combinations whose parameters violate the strategy's own validation (e.g.
    ``fast >= slow`` for SMA crossover) are silently skipped.

    Returns a DataFrame with one row per evaluated combination, plus the
    columns ``total_return_pct``, ``cagr_pct``, ``sharpe``, ``sortino``,
    ``max_drawdown_pct``, ``num_trades`` and ``volatility_pct``.
    """
    combos = _expand_grid(param_grid)
    rows: list[dict[str, Any]] = []
    for params in combos:
        try:
            strategy = build_strategy(strategy_name, dict(params))
        except (TypeError, ValueError) as exc:
            logger.debug(
                "Skipping invalid combo %s for %s: %s",
                _short_params(params),
                strategy_name,
                exc,
            )
            continue
        result = run_backtest(
            bars,
            strategy,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            slippage=slippage,
            interval=interval,
        )
        report = result.report
        row: dict[str, Any] = dict(params)
        row.update(
            {
                "total_return_pct": report.total_return_pct,
                "cagr_pct": report.cagr_pct,
                "sharpe": report.sharpe,
                "sortino": report.sortino,
                "max_drawdown_pct": report.max_drawdown_pct,
                "volatility_pct": report.volatility_pct,
                "num_trades": report.num_trades,
            }
        )
        rows.append(row)
    if not rows:
        param_cols = list(param_grid.keys())
        metric_cols = [
            "total_return_pct",
            "cagr_pct",
            "sharpe",
            "sortino",
            "max_drawdown_pct",
            "volatility_pct",
            "num_trades",
        ]
        return pd.DataFrame(columns=param_cols + metric_cols)
    return pd.DataFrame(rows)


def _split_windows(
    bars: pd.DataFrame, train_bars: int, test_bars: int, step_bars: int | None = None
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Slice ``bars`` into rolling (train, test) chunks of fixed bar counts."""
    n = len(bars)
    step = step_bars or test_bars
    if train_bars <= 0 or test_bars <= 0 or step <= 0:
        raise ValueError("train_bars, test_bars and step_bars must be positive")
    if n < train_bars + test_bars:
        raise ValueError(
            f"need at least {train_bars + test_bars} bars for one walk-forward "
            f"window (got {n})"
        )
    chunks: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    start = 0
    while start + train_bars + test_bars <= n:
        train = bars.iloc[start : start + train_bars]
        test = bars.iloc[start + train_bars : start + train_bars + test_bars]
        chunks.append((train, test))
        start += step
    return chunks


def walk_forward(
    bars: pd.DataFrame,
    strategy_name: str,
    param_grid: Mapping[str, Sequence[Any]],
    *,
    train_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    metric: str = "sharpe",
    initial_cash: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage: float = 0.0005,
    interval: str = "60",
) -> pd.DataFrame:
    """Walk-forward validation across rolling (train, test) windows.

    For each window the parameter grid is searched on the train slice, the
    combination with the highest ``metric`` is selected, and that single
    combination is then evaluated on the test slice. Both the in-sample
    (train) and out-of-sample (test) results are recorded.

    A strategy is only worth deploying live if ``test_<metric>`` is consistently
    positive across windows AND not catastrophically lower than ``train_<metric>``.
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"metric must be one of {SUPPORTED_METRICS}, got {metric!r}"
        )

    windows = _split_windows(bars, train_bars, test_bars, step_bars)
    combos = _expand_grid(param_grid)
    rows: list[dict[str, Any]] = []
    for window_idx, (train, test) in enumerate(windows):
        best_metric_value: float | None = None
        best_combo: dict[str, Any] | None = None
        best_train_report = None
        for combo in combos:
            try:
                strategy = build_strategy(strategy_name, dict(combo))
            except (TypeError, ValueError) as exc:
                logger.debug(
                    "Skipping invalid combo %s for %s: %s",
                    _short_params(combo),
                    strategy_name,
                    exc,
                )
                continue
            train_result = run_backtest(
                train,
                strategy,
                initial_cash=initial_cash,
                fee_rate=fee_rate,
                slippage=slippage,
                interval=interval,
            )
            metric_value = float(getattr(train_result.report, metric))
            if pd.isna(metric_value):
                continue
            if best_metric_value is None or metric_value > best_metric_value:
                best_metric_value = metric_value
                best_combo = dict(combo)
                best_train_report = train_result.report
        if best_combo is None or best_train_report is None:
            continue

        strategy = build_strategy(strategy_name, dict(best_combo))
        test_result = run_backtest(
            test,
            strategy,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            slippage=slippage,
            interval=interval,
        )
        test_report = test_result.report

        row: dict[str, Any] = {
            "window_idx": window_idx,
            "train_start": train.index[0],
            "train_end": train.index[-1],
            "test_start": test.index[0],
            "test_end": test.index[-1],
            "best_params": _short_params(best_combo),
        }
        for k, v in best_combo.items():
            row[f"best_{k}"] = v
        row[f"train_{metric}"] = float(getattr(best_train_report, metric))
        row["train_total_return_pct"] = best_train_report.total_return_pct
        row["train_max_drawdown_pct"] = best_train_report.max_drawdown_pct
        row[f"test_{metric}"] = float(getattr(test_report, metric))
        row["test_total_return_pct"] = test_report.total_return_pct
        row["test_max_drawdown_pct"] = test_report.max_drawdown_pct
        row["test_sharpe"] = test_report.sharpe
        row["test_num_trades"] = test_report.num_trades
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "SUPPORTED_METRICS",
    "parameter_sweep",
    "walk_forward",
]
