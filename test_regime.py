import sys, math
sys.path.insert(0, '.')
import numpy as np
from data.cache import DataCache
from indicators import add_all_indicators
from probability.signal_scorer import SignalScorer
from probability.momentum_scorer import MomentumScorer
from config.settings import (
    INITIAL_CAPITAL_USDT, MAX_RISK_PER_TRADE_PCT,
    SIGNAL_THRESHOLD, MOMENTUM_THRESHOLD, MOMENTUM_ADX_MIN,
    MOMENTUM_ATR_SL_MULT, MOMENTUM_ATR_TP_RATIO,
)

TAKER_FEE = 0.001

print("Loading data...")
cache = DataCache()

df_15m = cache.load('BTCUSDT', '15m')
df_15m = add_all_indicators(df_15m)
df_4h = cache.load('BTCUSDT', '4h')
df_4h = add_all_indicators(df_4h)

start = max(df_15m.index[0], df_4h.index[0])
end = min(df_15m.index[-1], df_4h.index[-1])
df_15m = df_15m[(df_15m.index >= start) & (df_15m.index <= end)]
df_4h = df_4h[(df_4h.index >= start) & (df_4h.index <= end)]
print(f"Range: {start} → {end}")

WARMUP = 300
cash = INITIAL_CAPITAL_USDT
trades = []

# ── 15m MR ─────────────────────────────────────────────────
print("Simulating 15m mean-reversion...")
mr = SignalScorer(long_threshold=SIGNAL_THRESHOLD)
i = WARMUP
while i < len(df_15m) - 1:
    row = df_15m.iloc[i]
    sig = mr.score(row)
    if not sig.is_tradeable() or sig.probability < SIGNAL_THRESHOLD:
        i += 1; continue

    entry = row["close"]
    atr = row.get("atr", float('nan'))
    if math.isnan(atr): atr = entry * 0.004
    sl_dist = atr * 1.25
    if sl_dist == 0: i += 1; continue
    sl = entry - sl_dist if sig.direction == "LONG" else entry + sl_dist
    tp = entry + sl_dist * 1.5 if sig.direction == "LONG" else entry - sl_dist * 1.5
    size = (cash * (MAX_RISK_PER_TRADE_PCT / 100)) / sl_dist
    if size <= 0: i += 1; continue

    pnl = None; ep = None
    for j in range(i + 1, min(i + 200, len(df_15m))):
        ch, cl = df_15m.iloc[j]["high"], df_15m.iloc[j]["low"]
        if sig.direction == "LONG":
            if cl <= sl: pnl, ep = (sl - entry) * size, sl; break
            if ch >= tp: pnl, ep = (tp - entry) * size, tp; break
        else:
            if ch >= sl: pnl, ep = (entry - sl) * size, sl; break
            if cl <= tp: pnl, ep = (entry - tp) * size, tp; break

    if pnl is not None:
        fee = (entry + ep) * size * TAKER_FEE; pnl -= fee; cash += pnl
        trades.append(("MR", sig.direction, pnl))
        i = j + 1
    else:
        i += 1

# ── 4h MOM ─────────────────────────────────────────────────
cash_mom = cash
print("Simulating 4h momentum...")
mom = MomentumScorer(
    long_threshold=MOMENTUM_THRESHOLD, momentum_adx_min=MOMENTUM_ADX_MIN,
    atr_sl_mult=MOMENTUM_ATR_SL_MULT, atr_tp_ratio=MOMENTUM_ATR_TP_RATIO,
)
i = WARMUP
while i < len(df_4h) - 1:
    row = df_4h.iloc[i]
    sig = mom.score(row)
    if not sig.is_tradeable() or sig.probability < MOMENTUM_THRESHOLD:
        i += 1; continue

    entry = row["close"]
    atr = row.get("atr", float('nan'))
    if math.isnan(atr): atr = entry * 0.004
    sl_dist = atr * MOMENTUM_ATR_SL_MULT
    if sl_dist == 0: i += 1; continue
    sl = entry - sl_dist if sig.direction == "LONG" else entry + sl_dist
    tp = entry + sl_dist * MOMENTUM_ATR_TP_RATIO if sig.direction == "LONG" else entry - sl_dist * MOMENTUM_ATR_TP_RATIO
    size = (cash_mom * (MAX_RISK_PER_TRADE_PCT / 100)) / sl_dist
    if size <= 0: i += 1; continue

    pnl = None; ep = None
    for j in range(i + 1, min(i + 200, len(df_4h))):
        ch, cl = df_4h.iloc[j]["high"], df_4h.iloc[j]["low"]
        if sig.direction == "LONG":
            if cl <= sl: pnl, ep = (sl - entry) * size, sl; break
            if ch >= tp: pnl, ep = (tp - entry) * size, tp; break
        else:
            if ch >= sl: pnl, ep = (entry - sl) * size, sl; break
            if cl <= tp: pnl, ep = (entry - tp) * size, tp; break

    if pnl is not None:
        fee = (entry + ep) * size * TAKER_FEE; pnl -= fee; cash_mom += pnl
        trades.append(("MOM", sig.direction, pnl))
        i = j + 1
    else:
        i += 1

# ── Results ──────────────────────────────────────────────────
print("\n" + "=" * 50)
print("COMBINED REGIME BACKTEST")
print("=" * 50)

trades_arr = np.array([t[2] for t in trades])
wins = trades_arr[trades_arr > 0]; losses = trades_arr[trades_arr < 0]
pf = round(sum(wins) / abs(sum(losses)), 2) if len(losses) else float('inf')
wr = round(len(wins) / len(trades) * 100, 1) if trades else 0

peak = INITIAL_CAPITAL_USDT; c = INITIAL_CAPITAL_USDT; max_dd = 0.0
for _, _, pnl in trades:
    c += pnl
    if c > peak: peak = c
    dd = (peak - c) / peak * 100
    if dd > max_dd: max_dd = dd

print(f"Total trades: {len(trades)}")
print(f"Win Rate: {wr}% ({len(wins)}W/{len(losses)}L)")
print(f"Profit Factor: {pf}")
print(f"Max Drawdown: {round(max_dd, 2)}%")
print(f"Net P&L: ${round(c - INITIAL_CAPITAL_USDT, 2)}")
print(f"Return: {round((c/INITIAL_CAPITAL_USDT-1)*100, 2)}%")

for strat in ["MR", "MOM"]:
    t = [x for x in trades if x[0] == strat]
    if t:
        a = np.array([x[2] for x in t])
        w = a[a > 0]; l = a[a < 0]
        spf = round(sum(w)/abs(sum(l)), 2) if len(l) else 'inf'
        swr = round(len(w)/len(t)*100, 1)
        net_s = round(sum(a), 2)
        print(f"  {strat}: {len(t):4d} trades, WR={swr:5.1f}%, PF={spf:6.2f}, Net=\${net_s:>8.2f}")
