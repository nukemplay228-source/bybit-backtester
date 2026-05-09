from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bybit_backtest.funding_arb import (
    FundingArbConfig,
    FundingArbResult,
    run_funding_arb,
)


def _make_8h_index(periods: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=periods, freq="8h", tz="UTC")


def _flat_klines(periods: int, price: float) -> pd.DataFrame:
    idx = _make_8h_index(periods)
    return pd.DataFrame(
        {
            "open": np.full(periods, price),
            "high": np.full(periods, price * 1.001),
            "low": np.full(periods, price * 0.999),
            "close": np.full(periods, price),
            "volume": np.full(periods, 100.0),
        },
        index=idx,
    )


def _flat_funding(periods: int, rate: float) -> pd.DataFrame:
    return pd.DataFrame({"funding_rate": np.full(periods, rate)}, index=_make_8h_index(periods))


def test_config_validates_margin_inequalities() -> None:
    with pytest.raises(ValueError):
        FundingArbConfig(notional=-1)
    with pytest.raises(ValueError):
        FundingArbConfig(initial_margin_rate=0.05, maintenance_margin_rate=0.10)
    with pytest.raises(ValueError):
        FundingArbConfig(maintenance_margin_rate=0)
    with pytest.raises(ValueError):
        FundingArbConfig(spot_fee_rate=-0.001)


def test_funding_arb_returns_funding_minus_fees_on_flat_market() -> None:
    spot = _flat_klines(30, 40000.0)
    perp = _flat_klines(30, 40000.0)
    funding = _flat_funding(30, 0.0001)
    config = FundingArbConfig(
        notional=1000.0,
        initial_margin_rate=0.33,
        maintenance_margin_rate=0.05,
        spot_fee_rate=0.0,
        perp_fee_rate=0.0,
        slippage=0.0,
    )

    result = run_funding_arb(spot, perp, funding, config)
    assert isinstance(result, FundingArbResult)
    assert not result.liquidated
    assert result.metrics["total_funding_received"] == pytest.approx(30 * 0.0001 * 1000.0, rel=1e-6)
    assert result.metrics["total_return"] > 0


def test_funding_arb_negative_rate_is_a_loss() -> None:
    spot = _flat_klines(20, 40000.0)
    perp = _flat_klines(20, 40000.0)
    funding = _flat_funding(20, -0.0002)
    config = FundingArbConfig(
        notional=1000.0,
        initial_margin_rate=0.33,
        spot_fee_rate=0.0,
        perp_fee_rate=0.0,
        slippage=0.0,
    )

    result = run_funding_arb(spot, perp, funding, config)
    assert result.metrics["total_funding_received"] < 0
    assert result.metrics["total_return"] < 0


def test_funding_arb_liquidates_on_adverse_perp_move_isolated() -> None:
    """Isolated-margin: a sharp perp move blows up the perp account
    even though the spot leg appreciates by the same amount."""
    spot = _flat_klines(5, 40000.0)
    perp_idx = _make_8h_index(5)
    perp = pd.DataFrame(
        {
            "open": [40000.0, 40000.0, 50000.0, 60000.0, 70000.0],
            "high": [40000.0, 40000.0, 50000.0, 60000.0, 70000.0],
            "low": [40000.0, 40000.0, 50000.0, 60000.0, 70000.0],
            "close": [40000.0, 40000.0, 50000.0, 60000.0, 70000.0],
            "volume": np.full(5, 100.0),
        },
        index=perp_idx,
    )
    funding = _flat_funding(5, 0.0001)
    config = FundingArbConfig(
        notional=1000.0,
        initial_margin_rate=0.10,
        maintenance_margin_rate=0.05,
        margin_mode="isolated",
    )

    result = run_funding_arb(spot, perp, funding, config)
    assert result.liquidated
    assert result.liquidation_time is not None


def test_funding_arb_cross_margin_survives_synced_pump() -> None:
    """Cross-margin: spot rallying alongside the perp keeps the account
    solvent even when the perp leg alone would have blown up."""
    idx = _make_8h_index(5)
    prices = [40000.0, 40000.0, 50000.0, 60000.0, 70000.0]
    spot = pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": np.full(5, 100.0),
        },
        index=idx,
    )
    perp = spot.copy()
    funding = _flat_funding(5, 0.0001)
    config = FundingArbConfig(
        notional=1000.0,
        initial_margin_rate=0.10,
        maintenance_margin_rate=0.05,
        margin_mode="cross",
    )

    result = run_funding_arb(spot, perp, funding, config)
    assert not result.liquidated


def test_funding_arb_rejects_empty_input() -> None:
    empty = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    funding = pd.DataFrame(
        {"funding_rate": pd.Series(dtype="float64")},
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    with pytest.raises(ValueError):
        run_funding_arb(empty, empty, funding)
