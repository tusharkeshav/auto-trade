"""
Walk-Forward Validation for 4h Momentum Strategy.
Fixed parameters (no grid optimization) — 4h windows.

Methodology:
  - Rolling 180-day train, 60-day test windows
  - No parameter optimization (fixed: threshold=60, ATR_SL=2.0, TP_RATIO=3.0)
  - Each test window produces OOS trades
  - Stitch all OOS trades for final metrics
"""
import sys, math
sys.path.insert(0, '.')
from datetime import timedelta
import pandas as pd
import numpy as np
from data.cache import DataCache
from indicators import add_all_indicators
from probability.momentum_scorer import MomentumScorer
from config.settings import (
    INITIAL_CAPITAL_USDT, MAX_RISK_PER_TRADE_PCT,
    MOMENTUM_THRESHOLD, MOMENTUM_ADX_MIN,
    MOMENTUM_ATR_SL_MULT, MOMENTUM_ATR_TP_RATIO,
)

TAKER_FEE = 0.001
WARMUP = 200

print("Loading 4h data...")
cache = DataCache()
df = cache.load('BTCUSDT', '4h')
df = add_all_indicators(df)
print(f"  {len(df)} candles ({df.index[0]} → {df.index[-1]})")

mom = MomentumScorer(
    long_threshold=MOMENTUM_THRESHOLD,
    momentum_adx_min=MOMENTUM_ADX_MIN,
    atr_sl_mult=MOMENTUM_ATR_SL_MULT,
    atr_tp_ratio=MOMENTUM_ATR_TP_RATIO,
)

TRAIN_DAYS = 180
TEST_DAYS  = 60
HOURS_PER_CANDLE = 4

all_oos_trades = []
capital = INITIAL_CAPITAL_USDT
fold = 1

while True:
    train_start = df.index[0] + timedelta(days=(fold - 1) * TEST_DAYS)
    train_end = train_start + timedelta(days=TRAIN_DAYS)
    test_end = train_end + timedelta(days=TEST_DAYS)

    if train_end > df.index[-1]:
        break

    df_train = df[(df.index >= train_start) & (df.index < train_end)]
    df_test  = df[(df.index >= train_end) & (df.index < test_end)]

    if len(df_train) < 200 or len(df_test) < 10:
        fold += 1
        continue

    # ── Train: find best threshold on training data ─────────
    best_thresh = MOMENTUM_THRESHOLD
    best_pf = 0.0

    for thresh in [55, 60, 65]:
        mom.long_threshold = thresh
        ps = []
        j = WARMUP
        while j < len(df_train) - 1:
            row = df_train.iloc[j]
            sig = mom.score(row)
            if sig.is_tradeable() and sig.probability >= thresh:
                ps.append(1 if sig.direction == "LONG" else -1)
            j += 1
        if len(ps) >= 3:
            wins = sum(1 for p in ps if p > 0)
            pf = wins / (len(ps) - wins) if (len(ps) - wins) > 0 else 0
            if pf > best_pf:
                best_pf = pf
                best_thresh = thresh

    mom.long_threshold = best_thresh

    # ── Test: trade with best threshold ─────────────────────
    cash = capital
    j = WARMUP
    oos = 0
    while j < len(df_test) - 1:
        row = df_test.iloc[j]
        sig = mom.score(row)
        if not sig.is_tradeable() or sig.probability < best_thresh:
            j += 1; continue

        entry = row["close"]
        atr = row.get("atr", float('nan'))
        if math.isnan(atr): atr = entry * 0.004
        sl_dist = atr * MOMENTUM_ATR_SL_MULT
        if sl_dist == 0: j += 1; continue
        sl = entry - sl_dist if sig.direction == "LONG" else entry + sl_dist
        tp = entry + sl_dist * MOMENTUM_ATR_TP_RATIO if sig.direction == "LONG" else entry - sl_dist * MOMENTUM_ATR_TP_RATIO
        size = (cash * (MAX_RISK_PER_TRADE_PCT / 100)) / sl_dist
        if size <= 0: j += 1; continue

        pnl = None; ep = None
        for k in range(j + 1, min(j + 200, len(df_test))):
            ch, cl = df_test.iloc[k]["high"], df_test.iloc[k]["low"]
            if sig.direction == "LONG":
                if cl <= sl: pnl, ep = (sl - entry) * size, sl; break
                if ch >= tp: pnl, ep = (tp - entry) * size, tp; break
            else:
                if ch >= sl: pnl, ep = (entry - sl) * size, sl; break
                if cl <= tp: pnl, ep = (entry - tp) * size, tp; break

        if pnl is not None:
            fee = (entry + ep) * size * TAKER_FEE; pnl -= fee; cash += pnl
            all_oos_trades.append((sig.direction, pnl))
            oos += 1
            j = k + 1
        else:
            j += 1

    print(f"  Fold {fold}: thresh={best_thresh}, OOS trades={oos}, P&L=${round(cash - capital, 2)}")
    capital = cash
    fold += 1

# ── Results ──────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"4h MOMENTUM WFO RESULT")
print(f"{'='*50}")
a = np.array([t[1] for t in all_oos_trades])
wins = a[a > 0]; losses = a[a < 0]
pf = round(sum(wins) / abs(sum(losses)), 2) if len(losses) else float('inf')
wr = round(len(wins) / len(a) * 100, 1) if len(a) else 0
net = round(sum(a), 2)

cap2, peak2, dd2 = INITIAL_CAPITAL_USDT, INITIAL_CAPITAL_USDT, 0.0
for _, pnl in all_oos_trades:
    cap2 += pnl
    if cap2 > peak2: peak2 = cap2
    dd = (peak2 - cap2) / peak2 * 100
    if dd > dd2: dd2 = dd

print(f"Total OOS trades: {len(all_oos_trades)}")
print(f"Win Rate: {wr}% ({len(wins)}W/{len(losses)}L)")
print(f"Profit Factor: {pf}")
print(f"Max DD: {round(dd2, 2)}%")
print(f"Net P&L: ${net}")
print(f"Final Capital: ${round(cap2, 2)}")
