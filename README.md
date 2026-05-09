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
  - `predictor` — multi-signal confluence (EMA trend + MACD + RSI + Donchian
    breakout + ATR volatility filter) with an ATR-based stop
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

### `predictor`

A "confluence" long-only strategy that combines five classical signals and
only takes a trade when **enough of them agree**. Exits on a trailing
ATR-based hard stop, an EMA trend break, or a MACD bearish cross — whichever
fires first.

The five signals (each contributes one vote, max score = 5):

1. **`trend_up`** — EMA(`fast_trend`) is above EMA(`slow_trend`).
2. **`macd_bullish`** — MACD line is above its signal line **and** the
   histogram is positive.
3. **`rsi_strong`** — RSI(`rsi_period`) is between `rsi_buy_min` and
   `rsi_buy_max` (default 50–70: strong but not yet overbought).
4. **`breakout`** — close is above the rolling Donchian high of the prior
   `donchian_period` bars.
5. **`volatility_ok`** — `ATR / close` is between `atr_pct_min` and
   `atr_pct_max` (skip dead and panic markets).

A long is opened when score ≥ `min_score`. The position is closed on the
first of:

- Close drops below `entry_price - atr_stop_mult × ATR_at_entry`
  (volatility-scaled hard stop).
- Close drops below the `fast_trend` EMA (trend-break exit).
- MACD line crosses below its signal line (momentum-flip exit).

| Param            | Type  | Default | Notes                                                                      |
|------------------|-------|---------|----------------------------------------------------------------------------|
| `fast_trend`     | int   | 50      | Fast EMA used for the trend filter and trend-break exit.                   |
| `slow_trend`     | int   | 200     | Slow EMA used for the trend filter. Must be > `fast_trend`.                |
| `macd_fast`      | int   | 12      | Fast EMA for the MACD line. Must be < `macd_slow`.                         |
| `macd_slow`      | int   | 26      | Slow EMA for the MACD line.                                                |
| `macd_signal`    | int   | 9       | EMA period applied to the MACD line itself.                                |
| `rsi_period`     | int   | 14      | RSI lookback (Wilder's smoothing).                                         |
| `rsi_buy_min`    | float | 50.0    | Lower bound of the "RSI strong" band.                                      |
| `rsi_buy_max`    | float | 70.0    | Upper bound of the "RSI strong" band.                                      |
| `donchian_period`| int   | 20      | Donchian breakout lookback (excluding the current bar).                    |
| `atr_period`     | int   | 14      | ATR lookback (Wilder's smoothing).                                         |
| `atr_pct_min`    | float | 0.002   | Minimum `ATR / close` to consider markets active.                          |
| `atr_pct_max`    | float | 0.10    | Maximum `ATR / close` to skip panic / illiquid spikes.                     |
| `atr_stop_mult`  | float | 2.0     | Hard stop = `entry - atr_stop_mult × ATR_at_entry`.                        |
| `min_score`      | int   | 4       | Minimum number of signals (out of 5) required to open a long. Range 1–5.   |

> [!CAUTION]
> The `predictor` strategy is **not** a guaranteed profit machine. It is a
> tighter filter than a single moving-average crossover; it will still take
> losing trades and will sit on the sidelines for long stretches in
> sideways markets. **Always** out-of-sample backtest your parameter choice
> on multiple symbols and time ranges before considering live capital.

## Live signal: `predict`

The `predict` subcommand downloads a window of the most recent klines for a
symbol and prints how the predictor's five signals line up on the most
recent closed bar — together with a suggested ATR-based stop and target if
the score is high enough.

```bash
python -m bybit_backtest predict \
    --symbol BTCUSDT \
    --interval 60 \
    --lookback-bars 500
```

Sample output:

```
Symbol      : BTCUSDT  (interval=60)
Last close  : 67421.500000  @ 2025-01-15T13:00:00+00:00

Indicators:
  EMA fast / slow   : 67110.214632 / 65902.418117
  RSI               : 58.42
  MACD / signal     : 312.214500 / 270.118300  (hist=+42.096200)
  ATR (1.21% of price): 815.620100
  Donchian hi / lo  : 67380.000000 / 65120.500000

Signals:
  trend_up      : True
  macd_bullish  : True
  rsi_strong    : True
  breakout      : True
  volatility_ok : True

Score       : 5 / 5  (confidence=100%)
Direction   : LONG
Suggested stop   : 65790.259800 (-2.0xATR from close)
Suggested target : 69868.360300 (+3.0xATR from close)

DISCLAIMER: this is a snapshot of indicator alignment, NOT a guaranteed
forecast. Past readings do not predict future returns; trade size is your
responsibility.
```

(Numbers above are illustrative; your run will differ.)

Pass `--params` to tune the predictor in-line, e.g.
`--params '{"min_score": 3, "atr_stop_mult": 2.5}'`. Pass `--json` to emit
the report as a single JSON object suitable for piping into other tools.

> [!CAUTION]
> The `predict` subcommand prints **diagnostic indicator readings**, not a
> high-probability forecast. No technical-analysis script can reliably
> predict short-term crypto price moves — anyone selling you one is wrong
> or lying. Treat the score as a discipline filter, never as a guarantee.

## Walk-forward validation & multi-asset scan

Picking the best `(fast, slow)` for SMA on a single year of BTC and then
"trusting" that result is the textbook way to fool yourself — you've selected
the parameters that look best **in hindsight**. Two CLI subcommands help guard
against that.

### `sweep` — walk-forward parameter search

Splits the price history into rolling **(train, test)** windows of fixed bar
counts. For each window, every combination from `--param-grid` is evaluated on
the **train** slice; the winner (by `--metric`) is then re-evaluated on the
**test** slice — bars the optimiser has never seen. The output table reports
both the in-sample and out-of-sample numbers, so the gap between them is
visible.

```bash
python -m bybit_backtest sweep \
    --strategy sma_cross \
    --symbol BTCUSDT --interval D \
    --start 2022-01-01 --end 2025-01-01 \
    --param-grid '{"fast": [5, 10, 20, 30, 50], "slow": [50, 100, 150, 200]}' \
    --train-bars 365 --test-bars 180 \
    --metric sharpe \
    --output-dir results/walk_forward
```

Example output for SMA crossover on BTC daily, 2022–2024 (train = 365 bars,
test = 180 bars, 4 walk-forward windows):

| Test window         | Best train params    | Train Sharpe | Test Sharpe | Test return | Test drawdown |
|---------------------|----------------------|-------------:|------------:|------------:|--------------:|
| 2023-01 → 2023-06   | fast=5,  slow=150    |        0.00  |       2.31  |     +15.43% |         2.00% |
| 2023-06 → 2023-12   | fast=10, slow=200    |        1.32  |       0.00  |       0.00% |         0.00% |
| 2023-12 → 2024-06   | fast=5,  slow=50     |        2.00  |      -0.31  |      -2.88% |         8.85% |
| 2024-06 → 2024-12   | fast=5,  slow=50     |        2.14  |       2.41  |     +48.56% |         8.39% |
| **Average**         |                      |       1.36   |       1.10  |     +15.28% |               |

Two of four out-of-sample windows are positive, one is flat (the strategy
never re-entered), one is a small loss. A sample of 4 is too small to draw
strong conclusions, but the procedure makes that explicit instead of hiding it
behind a single in-sample number.

### `scan` — multi-asset scanner

Runs the **same strategy with one fixed parameter set** across many symbols
and ranks them by Sharpe. A real edge should generalise across at least
several liquid pairs; an "edge" that only works on one symbol is usually
overfitting.

```bash
python -m bybit_backtest scan \
    --strategy sma_cross \
    --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,DOTUSDT,LINKUSDT,AVAXUSDT \
    --interval D \
    --start 2022-01-01 --end 2025-01-01 \
    --params '{"fast": 50, "slow": 200}' \
    --output-dir results/scanner
```

Example output for SMA(50, 200) daily, 2022–2024:

| Symbol   | Total return | Sharpe | Max drawdown | Trades |
|----------|-------------:|-------:|-------------:|-------:|
| BTCUSDT  |    +160.78%  |   1.06 |        26.1% |      2 |
| SOLUSDT  |    +143.97%  |   0.79 |        50.5% |      3 |
| DOGEUSDT |    +124.77%  |   0.74 |        52.7% |      2 |
| XRPUSDT  |     +66.03%  |   0.54 |        68.3% |      4 |
| BNBUSDT  |     +53.70%  |   0.55 |        43.8% |      3 |
| AVAXUSDT |     +42.86%  |   0.49 |        47.9% |      2 |
| ETHUSDT  |     +28.02%  |   0.40 |        43.0% |      2 |
| LINKUSDT |     +26.73%  |   0.40 |        46.2% |      3 |
| ADAUSDT  |      -2.20%  |   0.21 |        46.1% |      3 |
| DOTUSDT  |     -32.23%  |  -0.10 |        60.3% |      2 |

8/10 symbols profitable, mean return +61%, mean Sharpe +0.51 — i.e. SMA(50,
200) "works" across the broader market over this period, but the edge is
modest, max drawdowns are high (40–70% on alts), and DOT was a clean loss.
Compare these numbers to **buy-and-hold** of the same asset before you decide
the strategy is interesting.

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
  data_csv.py     # Public Bybit kline CSV archive loader (geo-block fallback)
  data_kucoin.py  # KuCoin spot/perp klines + funding-rate downloader
  portfolio.py    # Cash, position, fees, equity curve
  indicators.py   # Reusable indicator helpers (EMA, MACD, RSI, ATR, Donchian, ...)
  strategies/
    base.py       # Strategy ABC + Order / OrderSide / context
    sma_cross.py
    rsi_mr.py
    grid.py
    dca.py
    macd.py
    bollinger.py
    donchian.py
    predictor.py  # Multi-signal long-only confluence strategy
  predict.py      # Snapshot of `predictor` signals on the latest closed bar
  funding_arb.py  # Cash-and-carry funding-rate arbitrage simulator
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
