"""Multi-asset scanner.

Runs the same strategy with the same parameters across many symbols and
returns a ranked dataframe of results. The point is that an "edge" found on
BTC alone usually isn't an edge: it's a quirk of how BTC traded that year.
A real edge should show up — at least weakly — across many uncorrelated
markets. If a strategy makes money on BTC and ETH and SOL and BNB but loses
on most altcoins, that's still useful information; if it only wins on one
symbol, treat it as overfitting.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import pandas as pd

from bybit_backtest.backtest import run_backtest
from bybit_backtest.data_kucoin import fetch_spot_klines
from bybit_backtest.strategies import build_strategy

logger = logging.getLogger(__name__)


KlineFetcher = Callable[..., pd.DataFrame]


def _default_kucoin_fetcher(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch klines from KuCoin spot, accepting BTCUSDT-style symbols."""
    if "-" not in symbol:
        if symbol.endswith("USDT"):
            kucoin_symbol = f"{symbol[:-4]}-USDT"
        else:
            raise ValueError(
                f"Cannot map symbol {symbol!r} to a KuCoin spot pair. "
                "Pass a 'BASE-QUOTE' symbol (e.g. 'BTC-USDT') or use a "
                "USDT-quoted symbol like 'BTCUSDT'."
            )
    else:
        kucoin_symbol = symbol
    return fetch_spot_klines(
        kucoin_symbol, interval, start, end, use_cache=use_cache
    )


def scan_assets(
    symbols: Iterable[str],
    interval: str,
    start: str,
    end: str,
    strategy_name: str,
    params: Mapping[str, Any],
    *,
    initial_cash: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage: float = 0.0005,
    fetcher: KlineFetcher | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Run ``strategy_name`` with ``params`` on every symbol and rank results.

    Symbols that fail to load (empty data, network error) are reported with
    NaN metrics rather than crashing the whole scan, so a slow / blocked
    symbol can't take down the run.

    Returns a DataFrame sorted by Sharpe descending, with one row per symbol.
    """
    fetcher = fetcher or _default_kucoin_fetcher
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        try:
            bars = fetcher(symbol, interval, start, end, use_cache=use_cache)
        except Exception as exc:
            logger.warning("Skipping %s: data fetch failed (%s)", symbol, exc)
            rows.append(
                {
                    "symbol": symbol,
                    "num_bars": 0,
                    "total_return_pct": float("nan"),
                    "cagr_pct": float("nan"),
                    "sharpe": float("nan"),
                    "max_drawdown_pct": float("nan"),
                    "num_trades": 0,
                    "error": str(exc),
                }
            )
            continue
        if bars.empty or len(bars) < 2:
            logger.warning("Skipping %s: only %d bars returned", symbol, len(bars))
            rows.append(
                {
                    "symbol": symbol,
                    "num_bars": len(bars),
                    "total_return_pct": float("nan"),
                    "cagr_pct": float("nan"),
                    "sharpe": float("nan"),
                    "max_drawdown_pct": float("nan"),
                    "num_trades": 0,
                    "error": "no_data",
                }
            )
            continue

        try:
            strategy = build_strategy(strategy_name, dict(params))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot build strategy {strategy_name!r} with params {dict(params)}: {exc}"
            ) from exc
        result = run_backtest(
            bars,
            strategy,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            slippage=slippage,
            interval=interval,
        )
        report = result.report
        rows.append(
            {
                "symbol": symbol,
                "num_bars": len(bars),
                "total_return_pct": report.total_return_pct,
                "cagr_pct": report.cagr_pct,
                "sharpe": report.sharpe,
                "sortino": report.sortino,
                "max_drawdown_pct": report.max_drawdown_pct,
                "num_trades": report.num_trades,
                "win_rate_pct": report.win_rate_pct,
                "error": "",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(by="sharpe", ascending=False, na_position="last").reset_index(drop=True)
    return df


__all__ = ["scan_assets"]
