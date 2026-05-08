"""Smoke tests for the CLI parser and list command."""

from __future__ import annotations

from bybit_backtest.cli import _build_parser, _cmd_list


def test_parser_accepts_run_command() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "run",
            "--strategy",
            "sma_cross",
            "--symbol",
            "BTCUSDT",
            "--interval",
            "60",
            "--start",
            "2024-01-01",
            "--end",
            "2024-02-01",
        ]
    )
    assert args.command == "run"
    assert args.strategy == "sma_cross"


def test_list_strategies_prints(capsys) -> None:  # type: ignore[no-untyped-def]
    rc = _cmd_list(None)  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert rc == 0
    for name in ("sma_cross", "grid", "rsi_mr", "dca"):
        assert name in out
