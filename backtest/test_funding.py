"""
Backtest: Funding Rate Contrarian Strategy.

Logic:
  Entry: fundingRate > LONG_ENTRY_THRESH (retail over-leveraged short → LONG)
         fundingRate < SHORT_ENTRY_THRESH (retail over-leveraged long → SHORT)
  Exit: fundingRate returns to neutral zone, OR SL/TP hit, OR max hold

Key insight: Funding rate mean-reverts. When it hits extremes, the
positioned side is overcrowded → fade it.
"""
import sys, math
sys.path.insert(0, '/home/akhil/PycharmProjects/automate-trading')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from data.cache import DataCache
from data.binance_futures import BinanceFuturesClient
from indicators import add_all_indicators
from config.settings import INITIAL_CAPITAL_USDT, MAX_RISK_PER_TRADE_PCT

TAKER_FEE = 0.001

# ── Config ──────────────────────────────────────────────
LONG_ENTRY_THRESH  = -0.0005    # funding ≤ -0.05% → LONG (extreme bearish)
SHORT_ENTRY_THRESH = 0.0005     # funding ≥  0.05% → SHORT (extreme bullish)
EXIT_NEUTRAL_HIGH  = 0.0002     # exit LONG when funding ≥ this
EXIT_NEUTRAL_LOW   = -0.0002    # exit SHORT when funding ≤ this
ATR_SL_MULT        = 2.0        # stop loss = 2 × ATR
ATR_TP_RATIO       = 2.0        # take profit = 2 × risk
MAX_HOLD_CANDLES   = 96         # max 48 hours on 30min timeframe
CAPITAL            = INITIAL_CAPITAL_USDT

# ── Fetch data ──────────────────────────────────────────
print("Fetching funding rate history...")
fc = BinanceFuturesClient()
funding = fc.get_funding_rate_history_bulk(symbol="BTCUSDT")
if funding.empty:
    print("No funding data. Falling back to cached...")
    exit(1)

print(f"  Got {len(funding)} funding records")
print(f"  Range: {funding['fundingTime'].iloc[0]} → {funding['fundingTime'].iloc[-1]}")

# Fetch matching OHLCV for price data (4h to match funding rate's 8h cadence)
cache = DataCache()
df_spot = cache.load("BTCUSDT", "4h")
if df_spot is None:
    print("No 4h data. Download first.")
    exit(1)
df_spot = add_all_indicators(df_spot)
print(f"  OHLCV: {len(df_spot)} 4h candles")

# Align: for each funding record, find the nearest 4h candle
funding["fundingTime"] = pd.to_datetime(funding["fundingTime"])
funding = funding.set_index("fundingTime")

# Merge funding rate into OHLCV: forward-fill funding rate onto candle timestamps
df_spot["fundingRate"] = float('nan')
for ft in funding.index:
    # Find nearest 4h candle (within 2 hours)
    idx = df_spot.index.searchsorted(ft)
    if idx > 0:
        df_spot.loc[df_spot.index[idx-1], "fundingRate"] = funding.loc[ft, "fundingRate"]

# Forward fill funding rate until next funding event
df_spot["fundingRate"] = df_spot["fundingRate"].ffill().bfill()
print(f"  Merged funding into {len(df_spot)} candles")

# ── Trade simulation ────────────────────────────────────
WARMUP = 100
cash = CAPITAL
trades = []
i = WARMUP

while i < len(df_spot) - 1:
    row = df_spot.iloc[i]
    fr = row.get("fundingRate", 0)
    if math.isnan(fr):
        i += 1; continue

    # Check entry conditions
    entry_signal = None
    if fr <= LONG_ENTRY_THRESH:
        entry_signal = "LONG"
    elif fr >= SHORT_ENTRY_THRESH:
        entry_signal = "SHORT"

    if entry_signal is None:
        i += 1; continue

    entry = row["close"]
    atr = row.get("atr", float('nan'))
    if math.isnan(atr): atr = entry * 0.004

    sl_dist = atr * ATR_SL_MULT
    if sl_dist == 0: i += 1; continue

    if entry_signal == "LONG":
        sl = entry - sl_dist; tp = entry + sl_dist * ATR_TP_RATIO
    else:
        sl = entry + sl_dist; tp = entry - sl_dist * ATR_TP_RATIO

    size = (cash * (MAX_RISK_PER_TRADE_PCT / 100)) / sl_dist
    if size <= 0: i += 1; continue

    # Walk forward — check SL/TP AND funding rate exit
    pnl = None; ep = None; exit_reason = ""
    for j in range(i + 1, min(i + MAX_HOLD_CANDLES, len(df_spot))):
        ch, cl = df_spot.iloc[j]["high"], df_spot.iloc[j]["low"]
        fr_now = df_spot.iloc[j].get("fundingRate", 0)
        if math.isnan(fr_now): fr_now = 0

        # Check SL/TP first (conservative)
        if entry_signal == "LONG":
            if cl <= sl:
                pnl = (sl - entry) * size; ep = sl; exit_reason = "SL"
                break
            if ch >= tp:
                pnl = (tp - entry) * size; ep = tp; exit_reason = "TP"
                break
            # Funding normalization exit
            if fr_now >= EXIT_NEUTRAL_HIGH:
                exit_price = df_spot.iloc[j]["close"]
                pnl = (exit_price - entry) * size; ep = exit_price; exit_reason = "FUND_EXIT"
                break
        else:
            if ch >= sl:
                pnl = (entry - sl) * size; ep = sl; exit_reason = "SL"
                break
            if cl <= tp:
                pnl = (entry - tp) * size; ep = tp; exit_reason = "TP"
                break
            if fr_now <= EXIT_NEUTRAL_LOW:
                exit_price = df_spot.iloc[j]["close"]
                pnl = (entry - exit_price) * size; ep = exit_price; exit_reason = "FUND_EXIT"
                break

    if pnl is not None and ep is not None:
        fee = (entry + ep) * size * TAKER_FEE
        pnl -= fee; cash += pnl
        trades.append((entry_signal, pnl, exit_reason, fr))
        i = j + 1
    else:
        i += 1

# ── Results ─────────────────────────────────────────────
print(f"\n{'='*55}")
print("FUNDING RATE CONTRARIAN BACKTEST")
print(f"  Entry thresholds: LONG ≤ {LONG_ENTRY_THRESH}, SHORT ≥ {SHORT_ENTRY_THRESH}")
print(f"  SL: {ATR_SL_MULT}×ATR | TP: {ATR_TP_RATIO}R | Max hold: {MAX_HOLD_CANDLES} candles")
print(f"{'='*55}")

a = np.array([t[1] for t in trades])
wins = a[a > 0]; losses = a[a < 0]
pf = round(sum(wins) / abs(sum(losses)), 2) if len(losses) else float('inf')
wr = round(len(wins) / len(a) * 100, 1) if len(a) else 0
net = round(sum(a), 2)

c2, p2, dd2 = CAPITAL, CAPITAL, 0.0
for _, pnl, _, _ in trades:
    c2 += pnl
    if c2 > p2: p2 = c2
    dd = (p2 - c2) / p2 * 100
    if dd > dd2: dd2 = dd

print(f"\nTotal trades: {len(trades)}")
print(f"Win Rate: {wr}% ({len(wins)}W/{len(losses)}L)")
print(f"Profit Factor: {pf}")
print(f"Max DD: {round(dd2, 2)}%")
print(f"Net P&L: ${net}")
print(f"Final Capital: ${round(c2, 2)}")
print(f"Return: {round((c2/CAPITAL-1)*100, 2)}%")

# By direction
for direction in ["LONG", "SHORT"]:
    t = [x for x in trades if x[0] == direction]
    if t:
        a2 = np.array([x[1] for x in t])
        w2 = a2[a2 > 0]; l2 = a2[a2 < 0]
        spf = round(sum(w2)/abs(sum(l2)), 2) if len(l2) else 'inf'
        swr = round(len(w2)/len(t)*100, 1)
        exits = {}
        for _, _, er, _ in t:
            exits[er] = exits.get(er, 0) + 1
        print(f"\n{direction}: {len(t)} trades, WR={swr}%, PF={spf}")
        print(f"  Exits: {exits}")

# By funding rate bucket
print(f"\n--- Performance by funding rate at entry ---")
buckets = {
    "extreme (>0.1%)": lambda f: f > 0.001,
    "high (0.05-0.1%)": lambda f: 0.0005 < f <= 0.001,
    "moderate (0.02-0.05%)": lambda f: 0.0002 < f <= 0.0005,
    "neutral (±0.02%)": lambda f: -0.0002 <= f <= 0.0002,
    "moderate neg (-0.02 to -0.05%)": lambda f: -0.0005 <= f < -0.0002,
    "high neg (-0.05 to -0.1%)": lambda f: -0.001 <= f < -0.0005,
    "extreme neg (<-0.1%)": lambda f: f < -0.001,
}
for label, cond in buckets.items():
    t = [x for x in trades if cond(x[3])]
    if t:
        a3 = np.array([x[1] for x in t])
        w3 = a3[a3 > 0]; l3 = a3[a3 < 0]
        spf = round(sum(w3)/abs(sum(l3)), 2) if len(l3) else 'inf'
        swr = round(len(w3)/len(t)*100, 1)
        print(f"  {label:30s}: {len(t):3d} trades, WR={swr:5.1f}%, PF={spf}")
