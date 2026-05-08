"""Built-in trading strategies for the backtester."""

from bybit_backtest.strategies.base import Order, OrderSide, Strategy, StrategyContext
from bybit_backtest.strategies.dca import DCAStrategy
from bybit_backtest.strategies.grid import SpotGridStrategy
from bybit_backtest.strategies.rsi_mr import RSIMeanReversionStrategy
from bybit_backtest.strategies.sma_cross import SMACrossStrategy

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "sma_cross": SMACrossStrategy,
    "rsi_mr": RSIMeanReversionStrategy,
    "grid": SpotGridStrategy,
    "dca": DCAStrategy,
}


def build_strategy(name: str, params: dict) -> Strategy:
    """Construct a strategy by name with the given parameter dict."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy {name!r}. Available: {sorted(STRATEGY_REGISTRY)}"
        )
    cls = STRATEGY_REGISTRY[name]
    return cls(**params)


__all__ = [
    "STRATEGY_REGISTRY",
    "DCAStrategy",
    "Order",
    "OrderSide",
    "RSIMeanReversionStrategy",
    "SMACrossStrategy",
    "SpotGridStrategy",
    "Strategy",
    "StrategyContext",
    "build_strategy",
]
