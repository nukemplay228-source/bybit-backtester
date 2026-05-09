"""Smoke tests for built-in strategies via the backtest engine."""

from __future__ import annotations

import pandas as pd
import pytest

from bybit_backtest.backtest import run_backtest
from bybit_backtest.strategies import (
    BollingerBandsStrategy,
    DCAStrategy,
    DonchianBreakoutStrategy,
    MACDStrategy,
    RSIMeanReversionStrategy,
    SMACrossStrategy,
    SpotGridStrategy,
    build_strategy,
)


def test_build_strategy_unknown_name() -> None:
    with pytest.raises(ValueError):
        build_strategy("does_not_exist", {})


def test_sma_cross_invalid_params() -> None:
    with pytest.raises(ValueError):
        SMACrossStrategy(fast=50, slow=20)


def test_rsi_invalid_params() -> None:
    with pytest.raises(ValueError):
        RSIMeanReversionStrategy(period=1)
    with pytest.raises(ValueError):
        RSIMeanReversionStrategy(oversold=80, overbought=70)


def test_grid_invalid_params() -> None:
    with pytest.raises(ValueError):
        SpotGridStrategy(low=100, high=100, levels=10)
    with pytest.raises(ValueError):
        SpotGridStrategy(low=100, high=200, levels=1)


def test_dca_invalid_params() -> None:
    with pytest.raises(ValueError):
        DCAStrategy(interval_bars=0)
    with pytest.raises(ValueError):
        DCAStrategy(cash_per_buy=-1)


def test_macd_invalid_params() -> None:
    with pytest.raises(ValueError):
        MACDStrategy(fast=26, slow=12)
    with pytest.raises(ValueError):
        MACDStrategy(fast=0)
    with pytest.raises(ValueError):
        MACDStrategy(signal=-1)


def test_bollinger_invalid_params() -> None:
    with pytest.raises(ValueError):
        BollingerBandsStrategy(period=1)
    with pytest.raises(ValueError):
        BollingerBandsStrategy(num_std=0)


def test_donchian_invalid_params() -> None:
    with pytest.raises(ValueError):
        DonchianBreakoutStrategy(entry_period=1)
    with pytest.raises(ValueError):
        DonchianBreakoutStrategy(exit_period=1)


def test_sma_cross_runs_and_records_equity(crossing_bars: pd.DataFrame) -> None:
    strategy = SMACrossStrategy(fast=5, slow=20)
    result = run_backtest(crossing_bars, strategy, initial_cash=1000.0, fee_rate=0.001, slippage=0.0)
    assert len(result.equity) == len(crossing_bars)
    # Up trend in second half should let the strategy go long at least once.
    assert (result.trades["side"] == "BUY").sum() >= 1


def test_dca_makes_periodic_buys(crossing_bars: pd.DataFrame) -> None:
    strategy = DCAStrategy(interval_bars=24, cash_per_buy=10.0)
    result = run_backtest(crossing_bars, strategy, initial_cash=1000.0, fee_rate=0.001, slippage=0.0)
    n_buys = (result.trades["side"] == "BUY").sum()
    # 120 bars / 24 interval ≈ 5 buys (off-by-one acceptable)
    assert 4 <= n_buys <= 6
    assert (result.trades["side"] == "SELL").sum() == 0


def test_grid_oscillation_produces_round_trips(oscillating_bars: pd.DataFrame) -> None:
    strategy = SpotGridStrategy(low=92.0, high=108.0, levels=8)
    result = run_backtest(
        oscillating_bars,
        strategy,
        initial_cash=1000.0,
        fee_rate=0.0,
        slippage=0.0,
    )
    n_buys = (result.trades["side"] == "BUY").sum()
    n_sells = (result.trades["side"] == "SELL").sum()
    assert n_buys > 0
    assert n_sells > 0
    # In a clean sinusoid with no fees, the grid should not lose money.
    assert result.equity.iloc[-1] >= result.equity.iloc[0] - 1e-6


def test_rsi_runs_on_oscillating(oscillating_bars: pd.DataFrame) -> None:
    strategy = RSIMeanReversionStrategy(period=10, oversold=30, overbought=70)
    result = run_backtest(
        oscillating_bars,
        strategy,
        initial_cash=1000.0,
        fee_rate=0.0,
        slippage=0.0,
    )
    # We should see at least a couple of round trips on a sinusoid.
    assert len(result.trades) >= 2


def test_macd_takes_at_least_one_trade(crossing_bars: pd.DataFrame) -> None:
    strategy = MACDStrategy(fast=6, slow=15, signal=4)
    result = run_backtest(
        crossing_bars, strategy, initial_cash=1000.0, fee_rate=0.001, slippage=0.0
    )
    assert len(result.equity) == len(crossing_bars)
    # Crossing bars trend down then up — MACD should fire a buy.
    assert (result.trades["side"] == "BUY").sum() >= 1


def test_bollinger_round_trips_on_oscillating(oscillating_bars: pd.DataFrame) -> None:
    strategy = BollingerBandsStrategy(period=20, num_std=1.5)
    result = run_backtest(
        oscillating_bars, strategy, initial_cash=1000.0, fee_rate=0.0, slippage=0.0
    )
    assert (result.trades["side"] == "BUY").sum() >= 1
    assert (result.trades["side"] == "SELL").sum() >= 1


def test_donchian_breakout_buys_on_uptrend(crossing_bars: pd.DataFrame) -> None:
    strategy = DonchianBreakoutStrategy(entry_period=10, exit_period=5)
    result = run_backtest(
        crossing_bars, strategy, initial_cash=1000.0, fee_rate=0.001, slippage=0.0
    )
    # The clear up-trend in the second half should trigger at least one breakout.
    assert (result.trades["side"] == "BUY").sum() >= 1
