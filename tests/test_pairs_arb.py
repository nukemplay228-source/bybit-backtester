from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bybit_backtest.pairs_arb import (
    PairsArbConfig,
    PairsArbResult,
    run_pairs_arb,
)


def _idx(periods: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=periods, freq="8h", tz="UTC")


def _flat_klines(periods: int, price: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": np.full(periods, price),
            "high": np.full(periods, price * 1.001),
            "low": np.full(periods, price * 0.999),
            "close": np.full(periods, price),
            "volume": np.full(periods, 100.0),
        },
        index=_idx(periods),
    )


def _flat_funding(periods: int, rate: float) -> pd.DataFrame:
    return pd.DataFrame({"funding_rate": np.full(periods, rate)}, index=_idx(periods))


def _mean_reverting_pair(
    periods: int, base: float, *, amplitude: float, period_bars: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct A and B whose log-spread oscillates as a sine wave."""
    t = np.arange(periods)
    spread = amplitude * np.sin(2 * np.pi * t / period_bars)
    a_close = base * np.exp(spread / 2.0)
    b_close = base * np.exp(-spread / 2.0)
    a = pd.DataFrame(
        {"open": a_close, "high": a_close, "low": a_close, "close": a_close, "volume": 100.0},
        index=_idx(periods),
    )
    b = pd.DataFrame(
        {"open": b_close, "high": b_close, "low": b_close, "close": b_close, "volume": 100.0},
        index=_idx(periods),
    )
    return a, b


def test_config_validates_z_thresholds_and_margin() -> None:
    with pytest.raises(ValueError):
        PairsArbConfig(lookback=2)
    with pytest.raises(ValueError):
        PairsArbConfig(entry_z=1.0, exit_z=2.0)
    with pytest.raises(ValueError):
        PairsArbConfig(notional_per_leg=-1.0)
    with pytest.raises(ValueError):
        PairsArbConfig(initial_margin_rate=0.05, maintenance_margin_rate=0.10)
    with pytest.raises(ValueError):
        PairsArbConfig(fee_rate=-0.1)
    with pytest.raises(ValueError):
        PairsArbConfig(margin_mode="weird")


def test_pairs_arb_rejects_input_shorter_than_lookback() -> None:
    a = _flat_klines(15, 100.0)
    b = _flat_klines(15, 100.0)
    fa = _flat_funding(15, 0.0)
    fb = _flat_funding(15, 0.0)
    cfg = PairsArbConfig(lookback=30)
    with pytest.raises(ValueError):
        run_pairs_arb(a, b, fa, fb, cfg)


def test_pairs_arb_no_signal_on_flat_pair_stays_flat() -> None:
    """If both legs trade at constant prices the spread is zero so z is undefined / flat."""
    n = 80
    a = _flat_klines(n, 1000.0)
    b = _flat_klines(n, 100.0)
    fa = _flat_funding(n, 0.0)
    fb = _flat_funding(n, 0.0)
    cfg = PairsArbConfig(lookback=20, entry_z=2.0, exit_z=0.5, fee_rate=0.0, slippage=0.0)

    result = run_pairs_arb(a, b, fa, fb, cfg)
    assert isinstance(result, PairsArbResult)
    assert (result.position == 0).all()
    assert result.metrics["num_round_trips"] == 0
    assert result.metrics["total_return"] == pytest.approx(0.0, abs=1e-9)


def test_pairs_arb_round_trips_on_mean_reverting_spread() -> None:
    """A clean sinusoidal spread should trigger many entries and exits."""
    n = 240
    a, b = _mean_reverting_pair(n, base=1000.0, amplitude=0.10, period_bars=24)
    fa = _flat_funding(n, 0.0)
    fb = _flat_funding(n, 0.0)
    cfg = PairsArbConfig(
        lookback=20,
        entry_z=1.0,
        exit_z=0.2,
        notional_per_leg=1000.0,
        fee_rate=0.0,
        slippage=0.0,
    )

    result = run_pairs_arb(a, b, fa, fb, cfg)
    assert result.metrics["num_round_trips"] >= 3
    assert result.metrics["total_return"] > 0.0


def test_pairs_arb_records_funding_payments() -> None:
    """When the strategy is short the long-funded leg, funding flows into PnL."""
    n = 120
    a, b = _mean_reverting_pair(n, base=1000.0, amplitude=0.05, period_bars=20)
    fa = _flat_funding(n, 0.0001)
    fb = _flat_funding(n, 0.0001)
    cfg = PairsArbConfig(
        lookback=20,
        entry_z=1.0,
        exit_z=0.2,
        notional_per_leg=1000.0,
        fee_rate=0.0,
        slippage=0.0,
    )

    result = run_pairs_arb(a, b, fa, fb, cfg)
    position_array = result.position.to_numpy()
    funding_array = result.funding_payments.to_numpy()
    assert (position_array != 0).any()
    assert (funding_array != 0).any()

    open_indices = np.flatnonzero(position_array != 0)
    first_open = int(open_indices[0])
    assert (funding_array[:first_open] == 0.0).all()


def test_pairs_arb_isolated_margin_can_liquidate_on_runaway_spread() -> None:
    """In isolated margin a divergent spread eventually exhausts the perp margin
    on the losing leg even if the other leg gains."""
    n = 50
    idx = _idx(n)
    a_prices = np.linspace(1000.0, 5000.0, n)
    b_prices = np.linspace(1000.0, 200.0, n)
    a = pd.DataFrame(
        {"open": a_prices, "high": a_prices, "low": a_prices, "close": a_prices, "volume": 100.0},
        index=idx,
    )
    b = pd.DataFrame(
        {"open": b_prices, "high": b_prices, "low": b_prices, "close": b_prices, "volume": 100.0},
        index=idx,
    )
    fa = _flat_funding(n, 0.0)
    fb = _flat_funding(n, 0.0)
    cfg = PairsArbConfig(
        lookback=10,
        entry_z=1.0,
        exit_z=0.2,
        initial_margin_rate=0.05,
        maintenance_margin_rate=0.02,
        fee_rate=0.0,
        slippage=0.0,
        margin_mode="isolated",
    )

    result = run_pairs_arb(a, b, fa, fb, cfg)
    assert result.liquidated
    assert result.liquidation_time is not None
