# 🤖 AutoTrader — BTC Mean Reversion Paper Trading Bot

A fully automated, research-validated crypto paper trading bot built in Python.
Trades Bitcoin using a **mean reversion strategy** with institutional liquidity timing,
running on 15-minute and 30-minute candles simultaneously.

> **Status:** Paper trading (no real money). Validated on 5+ years of historical BTC data.

---

## Strategy Overview

The core idea: **BTC overshoots during low-ADX, high-liquidity sessions and reverts**.

The bot enters when:
1. ADX ≤ 25 (low trend strength — mean reversion conditions)
2. UTC 16:00–24:00 (institutional liquidity window — London/NY overlap + NY close)
3. Signal probability score ≥ 48 (composite scorer using RSI, Bollinger Bands, volume, S/R levels)

### Risk Management
- **All-In / All-Out** — 2% risk per trade, single TP at 1.5R, SL at 1.25×ATR
- Maximum 1 position per symbol at a time
- No overnight leverage, no margin

### Validated Performance (5-year backtest + OOS Walk-Forward)

| Metric | In-Sample | Out-of-Sample |
|---|---|---|
| Trades / year | 17.8 | 17.5 |
| Win Rate | 56.0% | 52.9% |
| Profit Factor | 1.91 | **1.69** |
| Max Drawdown | 8.7% | 10.1% |

> OOS = rolling 90-day train / 30-day test windows on unseen data.
> In-sample PF → OOS PF decay of only 0.22 indicates the edge is **real, not curve-fitted**.

---

## Architecture

```
automate-trading/
│
├── main.py                  # Entry point — live paper trading loop
│
├── config/
│   └── settings.py          # All strategy parameters (threshold, ADX, session, risk)
│
├── data/
│   ├── binance_client.py    # REST API wrapper (public endpoints only — no API key needed)
│   ├── cache.py             # Local OHLCV cache to avoid redundant API calls
│   └── historical/          # Downloaded CSV data for backtesting
│
├── indicators/              # Technical indicators (SMA, EMA, MACD, ADX, RSI, BB, ATR, VWAP, OBV)
│
├── probability/
│   └── signal_scorer.py     # Core scorer — combines all indicators into a 0–100 probability score
│
├── engine/
│   ├── order_manager.py     # Opens/closes paper positions, enforces risk rules
│   ├── portfolio.py         # Tracks open positions and P&L in real time
│   └── ledger.py            # Records all closed trades to Parquet on disk
│
├── dashboard/
│   ├── cli_dashboard.py     # Rich terminal dashboard
│   └── web_server.py        # Browser dashboard at http://localhost:8080
│
├── backtest/
│   ├── engines/standard.py  # Candle-by-candle backtest engine
│   └── engines/walk_forward.py  # Rolling OOS walk-forward optimizer
│
└── research scripts         # One-off validation and diagnostic scripts (see below)
```

---

## Quickstart

### 1. Clone and set up

```bash
git clone <repo-url>
cd automate-trading

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Verify connection to Binance (public API — no key required)

```bash
python test_binance.py
```

Expected output:
```
✅ Binance connection OK
BTC price: $XXXXX.XX
```

### 3. Run the live paper trading bot

```bash
python main.py
```

The bot will:
- Wait for the next 15-minute candle close
- Scan BTC on both 15m and 30m timeframes
- Open paper positions when the signal fires
- Monitor SL/TP every 30 seconds
- Display a live terminal dashboard

**Web dashboard** (auto-refreshes every 5s):
```
http://localhost:8080
```

**Stop the bot:** `Ctrl+C` — saves the trade ledger automatically.

---

## Configuration

All parameters are in `config/settings.py`. The critical ones:

```python
# Assets
SYMBOLS            = ["BTCUSDT", "ETHUSDT"]
SECONDARY_SYMBOLS  = ["BTCUSDT"]   # 30m scanning — BTC only (research-validated)

# Macro Shield — NEVER change these without re-running backtests
MACRO_MAX_ADX      = 25.0          # Skip trades when trend is strong
MACRO_SESSION_START = 16           # UTC 16:00 — London/NY open
MACRO_SESSION_END   = 24           # UTC 24:00 — NY close

# Entry
SIGNAL_THRESHOLD   = 48.0          # Minimum probability score to trade

# Risk
MAX_RISK_PER_TRADE_PCT = 2.0       # 2% of capital risked per trade
ATR_SL_MULT        = 1.25          # Stop-loss = 1.25 × ATR
ATR_TP_MULT        = 1.5           # Take-profit = 1.5 × risk (1.5R)

# Capital
INITIAL_CAPITAL_USDT = 10_000.0
```

> ⚠️ The **Macro Shield** (ADX + session window) is the core filter. These parameters are locked after extensive research. Loosening them reduces Profit Factor significantly.

---

## Getting Historical Data

All backtesting scripts read from local CSV files stored in `data/historical/`.
You need to download this data once before running any backtest.

### Download script

```bash
python download_data.py
```

This fetches OHLCV candles from Binance's **public REST API** (no API key needed)
and saves them as CSV files to `data/historical/`.

Expected output:
```
📥  Data Downloader — Fetching historical OHLCV

  ⬇  BTCUSDT  15m  — fetching 180,000 candles... ✅  180,000 rows  (2021-04-26 → 2026-06-14)
```

### Force re-download (if cache is stale)

```bash
python download_data.py --force
```

By default the script skips re-downloading if the file is less than 12 hours old.
Use `--force` to always refresh.

### Customising what to download

Edit the top of `download_data.py`:

```python
SYMBOLS   = ["BTCUSDT", "ETHUSDT"]   # add any Binance symbol
INTERVALS = ["15m", "30m"]            # candle timeframe
CANDLES   = 180000                    # 180k × 15m ≈ 5 years of data
```

| Candle count | Timeframe | Approximate history |
|---|---|---|
| 180,000 | 15m | ~5 years |
| 90,000 | 30m | ~5 years |
| 87,000 | 15m | ~2.5 years |
| 8,760 | 1h | ~1 year |

> **Note:** Binance rate-limits large fetches. The client handles pagination automatically — downloading 180k candles takes ~2–3 minutes.

### Where files are saved

```
data/historical/
  BTCUSDT_15m_180000.csv    # BTC 5-year 15m data  (~88 MB)
  ETHUSDT_15m_87000.csv     # ETH 2.5-year 15m data (~42 MB)
```

---

## Backtesting


This saves OHLCV CSVs to `data/historical/`.

### Run a standard backtest

```bash
python run_backtest.py
```

### Run Walk-Forward Out-of-Sample validation

```bash
python run_wfo.py
```

### Run the full combined 15m + 30m validation (recommended)

```bash
python run_combined_validation.py
```

This runs:
1. In-sample backtest (15m alone, 30m alone, combined)
2. OOS walk-forward (59 rolling 30-day test folds)
3. Final verdict table

---

## Research Scripts

These were used during strategy development. Run them to reproduce findings:

| Script | Purpose |
|---|---|
| `research_30m_standalone.py` | Validates 30m timeframe edge on BTC and ETH |
| `research_tf_overlap.py` | Measures signal independence between 15m and 30m |
| `research_dual_tf_backtest.py` | Combined 15m+30m simulation with position guard |
| `research_summary.py` | Go/No-Go verdict from all three research scripts |
| `diagnose_eth.py` | Deep ETH analysis — why 15m fails, 30m works |
| `deep_trade_analysis.py` | WFO trade slicing by ADX, hour, RSI, day-of-week |
| `monte_carlo.py` | Monte Carlo simulation of trade sequence risk |

---

## Trade Ledger

Every closed trade is saved to `data/ledger/` as a Parquet file.

To analyze past trades:

```bash
python analyze_trades.py
```

---

## How the Signal Scorer Works

The `SignalScorer` (`probability/signal_scorer.py`) takes a single OHLCV+indicators row and returns a probability score between 0–100.

It checks:
- **Macro Shield first** (hard gate): ADX and session window. Fails → `NO_TRADE` immediately.
- **RSI extremes** — oversold/overbought conditions
- **Bollinger Band position** — price at/outside bands
- **Volume confirmation** — above-average volume on the signal candle
- **Support/Resistance proximity** — signal fires near tested S/R levels
- **MACD and trend alignment** — ensures reversion direction is against the minor trend

Scores above `SIGNAL_THRESHOLD` (default 48.0) trigger an entry attempt.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| 15m + 30m BTC only | Research showed only 9% signal overlap — genuinely independent signals |
| ETH excluded from 30m | ETH 15m PF = 1.00 (breakeven). Only 10 ETH 30m trades in 2.5yr — not statistically significant |
| All-In / All-Out exit | Partial exits added complexity without improving OOS PF |
| ADX ≤ 25 locked | Loosening to ADX ≤ 30 reduced PF consistently across all test periods |
| 16–24 UTC window | Outside this window, institutional flow dominates and mean reversion fails |
| 2% risk per trade | Keeps max drawdown under 12% even during losing streaks |

---

## Requirements

```
Python 3.11+
pandas
numpy
requests
loguru
rich
fastapi
uvicorn
pyarrow
```

Install: `pip install -r requirements.txt`

No Binance API key required — the bot uses **public REST endpoints only** (no trading, no authentication).

---

## Disclaimer

This is a **paper trading research project**. It does not execute real trades or handle real money.
Past backtest performance does not guarantee future results.
Always do your own research before live trading.
