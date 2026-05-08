"""Command-line interface for running backtests.

Examples
--------
Run an SMA crossover on 1h BTC/USDT for 2024::

    python -m bybit_backtest run \
        --strategy sma_cross \
        --symbol BTCUSDT \
        --interval 60 \
        --start 2024-01-01 --end 2025-01-01 \
        --params '{"fast": 20, "slow": 50}'

List available strategies::

    python -m bybit_backtest list-strategies
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from bybit_backtest._version import __version__
from bybit_backtest.backtest import run_backtest
from bybit_backtest.data import load_klines
from bybit_backtest.data_csv import SUPPORTED_INTERVALS as CSV_SUPPORTED
from bybit_backtest.data_csv import load_klines_csv
from bybit_backtest.plotting import buy_and_hold_equity, plot_equity_curve
from bybit_backtest.strategies import STRATEGY_REGISTRY, build_strategy

logger = logging.getLogger("bybit_backtest")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bybit-backtest",
        description="Backtest trading strategies on Bybit historical kline data.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase log verbosity (-v, -vv)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a backtest.")
    run.add_argument("--strategy", required=True, choices=sorted(STRATEGY_REGISTRY))
    run.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    run.add_argument(
        "--interval",
        required=True,
        help="Bybit interval: 1, 5, 15, 30, 60, 240, 720, D, W, M, ...",
    )
    run.add_argument("--start", required=True, help="UTC start date, YYYY-MM-DD")
    run.add_argument("--end", required=True, help="UTC end date, YYYY-MM-DD (exclusive)")
    run.add_argument(
        "--category",
        default="spot",
        choices=["spot", "linear", "inverse"],
        help="Bybit market category (default: spot). Only applies when --source=api.",
    )
    run.add_argument(
        "--source",
        default="api",
        choices=["api", "public_csv"],
        help=(
            "Where to fetch klines from. 'api' uses api.bybit.com (default, all intervals). "
            "'public_csv' uses the static CSV archives at public.bybit.com — useful when "
            "api.bybit.com is geo-blocked, but only supports 1/5/15/30/60-minute intervals."
        ),
    )
    run.add_argument(
        "--initial-cash", type=float, default=10_000.0, help="Initial cash in USDT (default: 10000)."
    )
    run.add_argument(
        "--fee-rate",
        type=float,
        default=0.001,
        help="Per-trade fee as a decimal (default: 0.001 = 0.1%%).",
    )
    run.add_argument(
        "--slippage",
        type=float,
        default=0.0005,
        help="One-sided slippage as a decimal (default: 0.0005 = 5 bps).",
    )
    run.add_argument(
        "--params",
        default="{}",
        help="JSON dict of strategy parameters, e.g. '{\"fast\": 10, \"slow\": 30}'",
    )
    run.add_argument(
        "--output-dir",
        default="results",
        help="Directory where reports and equity curve plots are written.",
    )
    run.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the on-disk kline cache and re-download from Bybit.",
    )
    run.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip drawing the buy-and-hold benchmark on the chart.",
    )

    sub.add_parser("list-strategies", help="List built-in strategies.")
    return parser


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        params = json.loads(args.params) if args.params else {}
    except json.JSONDecodeError as exc:
        print(f"Invalid --params JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(params, dict):
        print("--params must decode to a JSON object", file=sys.stderr)
        return 2

    strategy = build_strategy(args.strategy, params)
    logger.info("Built strategy %s with params %s", strategy.name, strategy.describe())

    if args.source == "public_csv":
        if args.interval not in CSV_SUPPORTED:
            print(
                f"--source=public_csv only supports intervals {sorted(CSV_SUPPORTED)}; "
                f"got {args.interval!r}.",
                file=sys.stderr,
            )
            return 2
        bars = load_klines_csv(
            args.symbol,
            args.interval,
            args.start,
            args.end,
            use_cache=not args.no_cache,
        )
    else:
        bars = load_klines(
            args.symbol,
            args.interval,
            args.start,
            args.end,
            category=args.category,
            use_cache=not args.no_cache,
        )
    if bars.empty:
        print(
            f"No klines returned for {args.symbol} {args.interval} {args.start}->{args.end}",
            file=sys.stderr,
        )
        return 1
    logger.info("Loaded %d bars (%s -> %s)", len(bars), bars.index[0], bars.index[-1])

    result = run_backtest(
        bars,
        strategy,
        initial_cash=args.initial_cash,
        fee_rate=args.fee_rate,
        slippage=args.slippage,
        interval=args.interval,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.strategy}_{args.symbol}_{args.interval}_{args.start}_{args.end}"
    safe_tag = tag.replace(":", "-").replace(" ", "_")

    trades_path = output_dir / f"{safe_tag}_trades.csv"
    equity_path = output_dir / f"{safe_tag}_equity.csv"
    report_path = output_dir / f"{safe_tag}_report.json"
    chart_path = output_dir / f"{safe_tag}_equity.png"

    result.trades.to_csv(trades_path, index=False)
    result.equity.to_csv(equity_path, header=["equity"])
    report_path.write_text(json.dumps(result.report.to_dict(), indent=2, default=str))

    benchmark = (
        None if args.no_benchmark else buy_and_hold_equity(bars, args.initial_cash, args.fee_rate)
    )
    plot_equity_curve(
        result.equity,
        chart_path,
        title=f"{args.strategy} {args.symbol} {args.interval} ({args.start} -> {args.end})",
        benchmark=benchmark,
    )

    print()
    print(f"Strategy : {strategy.name}")
    print(f"Params   : {json.dumps(strategy.describe())}")
    print(
        f"Symbol   : {args.symbol}  interval={args.interval}  "
        f"{args.start} -> {args.end}  ({len(bars)} bars)"
    )
    print()
    print(result.report.to_text())
    print()
    print(f"Trades CSV : {trades_path}")
    print(f"Equity CSV : {equity_path}")
    print(f"Report JSON: {report_path}")
    print(f"Chart      : {chart_path}")
    return 0


def _cmd_list(_: argparse.Namespace) -> int:
    for name, cls in sorted(STRATEGY_REGISTRY.items()):
        doc = (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""
        print(f"  {name:<10}  {doc}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "list-strategies":
        return _cmd_list(args)
    parser.error(f"Unknown command: {args.command}")
    return 2  # unreachable but keeps mypy happy


if __name__ == "__main__":
    raise SystemExit(main())
