#!/usr/bin/env bash
# Runnable examples for the bybit-backtester. Each command writes its
# output into ./results/.
#
# Usage:
#   ./examples/run_examples.sh
#
# Adjust the date range / symbols freely. Bybit's public kline endpoint is
# free and does not require an API key.

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p results

# 1. SMA crossover on 1h BTC/USDT for 2024.
python -m bybit_backtest run \
    --strategy sma_cross \
    --symbol BTCUSDT \
    --interval 60 \
    --start 2024-01-01 --end 2025-01-01 \
    --params '{"fast": 20, "slow": 50}'

# 2. RSI mean-reversion on 4h ETH/USDT for 2024.
python -m bybit_backtest run \
    --strategy rsi_mr \
    --symbol ETHUSDT \
    --interval 240 \
    --start 2024-01-01 --end 2025-01-01 \
    --params '{"period": 14, "oversold": 30, "overbought": 70}'

# 3. Spot grid on 1h SOL/USDT in a 2024 range.
python -m bybit_backtest run \
    --strategy grid \
    --symbol SOLUSDT \
    --interval 60 \
    --start 2024-06-01 --end 2024-12-01 \
    --params '{"low": 120, "high": 200, "levels": 20}'

# 4. DCA benchmark on daily BTC/USDT for 2024 — buys $50 every 7 bars (~weekly).
python -m bybit_backtest run \
    --strategy dca \
    --symbol BTCUSDT \
    --interval D \
    --start 2024-01-01 --end 2025-01-01 \
    --params '{"interval_bars": 7, "cash_per_buy": 50}'
