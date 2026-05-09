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
  - `macd` — MACD signal-line crossover
  - `bollinger` — Bollinger Bands mean reversion
  - `donchian` — Donchian channel breakout (Turtle-style)
- **Funding-rate arbitrage backtest** (cash-and-carry: long spot + short
  perpetual, delta-neutral, collect funding payments). See the
  [Funding-rate arbitrage](#funding-rate-arbitrage-cash-and-carry) section.
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

### `macd`

| Param    | Type | Default | Notes                                                    |
|----------|------|---------|----------------------------------------------------------|
| `fast`   | int  | 12      | Fast EMA period. Must be < `slow`.                       |
| `slow`   | int  | 26      | Slow EMA period.                                         |
| `signal` | int  | 9       | EMA period applied to the MACD line itself.              |

### `bollinger`

| Param     | Type  | Default | Notes                                          |
|-----------|-------|---------|------------------------------------------------|
| `period`  | int   | 20      | Lookback window for the moving average / std.  |
| `num_std` | float | 2.0     | Width of the bands in standard deviations.     |

### `donchian`

| Param           | Type | Default | Notes                                                  |
|-----------------|------|---------|--------------------------------------------------------|
| `entry_period`  | int  | 55      | Breakout lookback. Buy on close > prior N highs.       |
| `exit_period`   | int  | 20      | Exit lookback. Sell on close < prior M lows.           |

## Funding-rate arbitrage (cash-and-carry)

The `funding-arb` subcommand simulates the classic delta-neutral
cash-and-carry trade: **buy 1 unit of spot** and **simultaneously short 1
unit of the same asset on a USDT-margined perpetual**. Spot gains and perp
losses (or vice versa) cancel out, so the only meaningful P&L is the funding
payment paid every 8 hours from longs to shorts (positive funding) or from
shorts to longs (negative funding) — minus fees and slippage.

Funding rates on Bybit / KuCoin / Binance are **public information** —
arbitrageurs close the gap between exchanges, so KuCoin's funding rate is a
reasonable proxy for Bybit's. The backtester uses **KuCoin** public data
(spot klines, perp klines, funding-rate history) because KuCoin is reachable
from most cloud regions while `api.bybit.com` is geo-blocked from many.

```bash
python -m bybit_backtest funding-arb \
    --symbol BTC \
    --start 2024-01-01 --end 2025-01-01 \
    --notional 1000 \
    --initial-margin-rate 0.33 \
    --maintenance-margin-rate 0.05
```

The simulator opens both legs on bar 0, posts initial margin on the perp
leg, marks-to-market every 8 hours (the funding settlement cadence), credits
funding payments to the perp account, and closes both legs on the last bar.
Liquidation is checked at every settlement; by default it uses **cross
margin** — i.e. the spot leg's market value is treated as collateral for the
perp short, which is how unified accounts on Bybit / KuCoin behave. Pass
`--margin-mode isolated` to model an isolated-margin perp account (which can
liquidate even on a delta-neutral position if the perp leg moves against you
faster than you can top up).

### Backtest results — BTC & ETH

Run on KuCoin 8h data, $1000 per leg, 33% initial margin, 5% maintenance
margin, 0.10% spot taker fee, 0.06% perp taker fee, 5 bps slippage per side,
cross margin:

| Period            | Total return | Annualised | Funding (USD) | Fees (USD) | Max drawdown | Liquidated |
|-------------------|--------------|------------|---------------|-----------|--------------|------------|
| BTC 2022 (bear)   | +1.16%       | +1.17%     | +18.58        | 2.17      | -0.59%       | no         |
| BTC 2023 (mixed)  | +16.76%      | +16.79%    | +233.43       | 5.69      | -0.45%       | no         |
| BTC 2024 (bull)   | +21.77%      | +21.75%    | +299.13       | 5.14      | -0.48%       | no         |
| ETH 2024 (bull)   | +20.62%      | +20.59%    | +281.04       | 3.94      | -0.26%       | no         |

CSV summary and PNG charts are written under `results/funding_arb/` after
running `python build_arb_summary.py`.

### What the numbers mean (in plain English)

- **In bull markets** (2023–2024) the trade earned **~17–22% per year**,
  about **1.4–1.8% per month**, paid in roughly steady 8h drips with very
  small drawdowns (<0.6% of capital deployed).
- **In a bear market** (2022) funding rates collapsed (longs aren't paying
  premium when nobody is leveraged-long), and the trade returned a flat
  **~1% per year** — barely above zero, definitely not a passive-income
  machine.
- The strategy is **not magic** — it requires real capital on **two**
  exchanges (or a unified account), real fees, and real margin
  monitoring. A 50% spike in basis or a single mistimed top-up can wipe
  weeks of funding.
- Numbers are based on **KuCoin** funding history. Bybit's rates are
  similar but not identical; expect ±2–3 percentage points in either
  direction.

### Caveats

- **Past funding ≠ future funding.** Rates compress during bears and
  during periods of low retail leverage.
- **Execution risk.** Real-life entry/exit at the same price on two
  venues is harder than in the backtest.
- **Counterparty risk.** Holding capital on a centralized exchange
  always carries platform-risk (FTX, etc.).
- **No transfers / borrowing modelled.** If you need to move USDT
  between spot and futures accounts there can be small fees and delays.

The conclusion: funding-rate arb is one of the more realistic ways to earn
**low-double-digit annualised returns** on crypto with **delta-neutral
exposure**. It's not "stable monthly income" — bear markets cut returns to
near-zero — but it's far closer to that ideal than directional trading.

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
    macd.py
    bollinger.py
    donchian.py
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
