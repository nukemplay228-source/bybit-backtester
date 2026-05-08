# bybit-backtester

A small, beginner-friendly Python backtester for crypto trading strategies on
[Bybit](https://www.bybit.com/) historical kline (candlestick) data. It downloads
public OHLCV bars, runs your strategy against them, and produces honest
performance metrics — accounting for trading fees and slippage — so you can
sanity-check an idea **before** risking any money.

> [!CAUTION]
> **This is an educational tool, not financial advice.** Backtest results are
> based on past data and DO NOT predict future returns. Most retail crypto
> traders lose money. Never trade with money you cannot afford to lose. Be
> especially skeptical of any strategy that "promises" double-digit monthly
> returns — they don't exist for free.

## Features

- **Public data**, no API keys: pulls klines from Bybit's V5 public endpoint
  (`/v5/market/kline`) and caches them on disk as parquet.
- **Long-only spot** portfolio with explicit fees, slippage, and
  mark-to-market equity tracking.
- **No look-ahead**: strategies make decisions on bar close; orders execute on
  the next bar's open.
- **Built-in strategies**:
  - `sma_cross` — moving average crossover
  - `rsi_mr` — RSI mean reversion (Wilder's smoothing)
  - `grid` — spot grid bot (pre-defined price range)
  - `dca` — dollar-cost averaging (passive benchmark)
- **Honest metrics**: total return, CAGR, max drawdown, Sharpe, Sortino,
  volatility, win rate, profit factor, average trade.
- **Outputs**: trades CSV, equity CSV, JSON report, equity-curve PNG (with a
  buy-and-hold benchmark overlaid).
- **CLI**: `python -m bybit_backtest run ...` or `bybit-backtest run ...`.

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/nukemplay228-source/bybit-backtester.git
cd bybit-backtester
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

Run an SMA crossover on hourly BTC/USDT spot for one year:

```bash
python -m bybit_backtest run \
    --strategy sma_cross \
    --symbol BTCUSDT \
    --interval 60 \
    --start 2024-01-01 --end 2025-01-01 \
    --params '{"fast": 20, "slow": 50}' \
    --initial-cash 10000 \
    --fee-rate 0.001 \
    --slippage 0.0005
```

Output files are written to `results/`:

```
results/
  sma_cross_BTCUSDT_60_2024-01-01_2025-01-01_trades.csv
  sma_cross_BTCUSDT_60_2024-01-01_2025-01-01_equity.csv
  sma_cross_BTCUSDT_60_2024-01-01_2025-01-01_report.json
  sma_cross_BTCUSDT_60_2024-01-01_2025-01-01_equity.png
```

The console summary looks like:

```
Strategy : sma_cross
Params   : {"name": "sma_cross", "fast": 20, "slow": 50}
Symbol   : BTCUSDT  interval=60  2024-01-01 -> 2025-01-01  (8784 bars)

Initial equity      10,000.00
Final equity        12,418.30
Total return        +24.18%
CAGR                +24.18%
Max drawdown        -11.42%
Volatility (ann.)   38.94%
Sharpe              0.71
Sortino             1.04
Trades              17
Win rate            41.18%
Profit factor       1.83
Avg trade           +1.42%

Trades CSV : results/sma_cross_BTCUSDT_60_2024-01-01_2025-01-01_trades.csv
Equity CSV : results/sma_cross_BTCUSDT_60_2024-01-01_2025-01-01_equity.csv
Report JSON: results/sma_cross_BTCUSDT_60_2024-01-01_2025-01-01_report.json
Chart      : results/sma_cross_BTCUSDT_60_2024-01-01_2025-01-01_equity.png
```

(Numbers above are illustrative; your run will differ.)

### List strategies

```bash
python -m bybit_backtest list-strategies
```

## Strategy parameters

Pass strategy parameters as a JSON dict via `--params`.

### `sma_cross`

| Param  | Type | Default | Notes                                  |
|--------|------|---------|----------------------------------------|
| `fast` | int  | 20      | Fast SMA window (in bars). Must be < `slow`. |
| `slow` | int  | 50      | Slow SMA window (in bars).             |

### `rsi_mr`

| Param         | Type  | Default | Notes                              |
|---------------|-------|---------|------------------------------------|
| `period`      | int   | 14      | RSI lookback (Wilder's smoothing). |
| `oversold`    | float | 30.0    | Buy threshold.                     |
| `overbought`  | float | 70.0    | Sell threshold.                    |

### `grid`

| Param               | Type  | Default | Notes                                            |
|---------------------|-------|---------|--------------------------------------------------|
| `low`               | float | —       | Bottom of grid range (USD).                      |
| `high`              | float | —       | Top of grid range (USD). Must be > `low`.        |
| `levels`            | int   | 20      | Number of grid lines.                            |
| `capital_fraction`  | float | 1.0     | Fraction of equity used by the grid (0-1).       |

### `dca`

| Param            | Type  | Default | Notes                                  |
|------------------|-------|---------|----------------------------------------|
| `interval_bars`  | int   | 24      | Buy every N bars.                      |
| `cash_per_buy`   | float | 10.0    | Cash spent on each buy.                |

## How the engine works

1. **Bars in chronological order.** Each bar contains `open, high, low, close,
   volume`. The strategy's `on_bar` is called with the bar (whose close is
   considered "now") and a snapshot of the portfolio state.
2. **Strategy returns orders** — `BUY` (with `cash_fraction` of available
   cash) or `SELL` (with `position_fraction` of the open position). Orders
   are queued for the **next** bar.
3. **Execution** happens at the next bar's open price, with a one-sided
   `slippage` adjustment (default 5 bps) and a per-trade `fee_rate`
   (default 10 bps to match Bybit spot taker).
4. **Equity is recorded** at every bar close (mark-to-market). This is what
   the Sharpe/volatility/drawdown metrics consume.

This setup avoids the most common backtest mistake — using same-bar high/low
for entries — which dramatically inflates apparent returns.

## Caveats

- **Spot only, long-only.** No shorts, no margin, no derivatives. Adding
  futures with funding and liquidation logic is outside this project's scope.
- **Bar-level execution.** Intra-bar fills are not simulated. Strategies that
  rely on tight stop placement or partial fills will not be modelled correctly.
- **Bybit's history depth varies by symbol.** Old, delisted, or recently
  added pairs may have less data than expected. The CLI will print the
  number of bars actually returned.
- **Fees default to taker.** Set `--fee-rate 0` to see a "no friction"
  upper bound, but always also run with realistic fees.
- **The grid strategy is in-memory.** It models a grid as logical slots
  rather than persistent on-exchange limit orders. For a real grid bot you
  would need to send and replace limit orders on the venue.

## Project layout

```
src/bybit_backtest/
  data.py         # Bybit V5 kline downloader + parquet cache
  portfolio.py    # Cash, position, fees, equity curve
  strategies/
    base.py       # Strategy ABC + Order / OrderSide / context
    sma_cross.py
    rsi_mr.py
    grid.py
    dca.py
  metrics.py      # Performance metrics (return, drawdown, Sharpe, ...)
  backtest.py     # Bar-by-bar engine
  plotting.py     # Equity curve + drawdown chart
  cli.py          # argparse-based CLI

tests/            # pytest suite — strategies, engine, metrics, data, CLI
```

## Development

```bash
# Run tests
pytest

# Lint
ruff check .

# Format check (optional, ruff formatter)
ruff format --check .
```

## License

MIT.
