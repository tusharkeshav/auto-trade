"""
deep_trade_analysis.py

10x Critical Analysis — find the ACTUAL edge in the raw trade data.
Runs WFO, captures every trade, then slices:
  - PF by hour of day
  - PF by day of week
  - PF by entry probability score bucket
  - PF by ATR percentile at entry
  - PF by ADX at entry
  - Win/loss streaks and clustering
  - Are ANY conditions producing PF > 1.5?
"""
import sys, math, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from backtest.engines.walk_forward import WalkForwardOptimizer
from data.cache import DataCache
from indicators import add_all_indicators

# ──────────────────────────────────────────────────────
# 1. Run WFO and collect ALL trades
# ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("  DEEP TRADE ANALYSIS — Finding the Real Edge")
print("="*60)

wfo = WalkForwardOptimizer(symbol="ETHUSDT", interval="15m", train_days=180, test_days=60)
result = wfo.run()
trades = result.trades

if not trades:
    print("No trades generated. Check filters.")
    sys.exit(1)

print(f"\n  Total OOS trades captured: {len(trades)}")

# ──────────────────────────────────────────────────────
# 2. Build a rich DataFrame of trades with entry context
# ──────────────────────────────────────────────────────
# Load the full indicator dataframe to join with entry conditions
cache = DataCache()
df = cache.load("ETHUSDT", "15m")
df = add_all_indicators(df)

rows = []
for t in trades:
    entry_time = t.entry_time
    if entry_time not in df.index:
        # try nearest
        idx = df.index.searchsorted(entry_time)
        if idx >= len(df): continue
        entry_time = df.index[idx]

    row = df.loc[entry_time]
    pnl = t.pnl
    is_win = pnl > 0
    r_multiple = pnl / (abs(t.pnl) if pnl < 0 else t.pnl) if pnl != 0 else 0
    # Actual R: pnl / (entry * size * 0.02) approximate — use exit type
    rows.append({
        "entry_time":    t.entry_time,
        "exit_type":     t.exit_type,
        "pnl":           pnl,
        "is_win":        is_win,
        "probability":   t.probability,
        "candles_held":  t.candles_held,
        "direction":     t.direction,
        "hour":          t.entry_time.hour,
        "dow":           t.entry_time.weekday(),  # 0=Mon, 6=Sun
        "adx":           row.get("adx", np.nan),
        "atr_pct":       row.get("atr_percentile", np.nan),
        "bb_pct":        row.get("bb_pct", np.nan),
        "rsi":           row.get("rsi", np.nan),
        "volume_ratio":  row.get("volume_ratio", np.nan),
    })

tdf = pd.DataFrame(rows)
print(f"  Enriched with market context for {len(tdf)} trades\n")

def pf(df_sub):
    wins = df_sub[df_sub["pnl"] > 0]["pnl"].sum()
    losses = abs(df_sub[df_sub["pnl"] < 0]["pnl"].sum())
    if losses == 0: return float("inf")
    return round(wins / losses, 2)

def wr(df_sub):
    if len(df_sub) == 0: return 0
    return round(df_sub["is_win"].mean() * 100, 1)

def show_table(title, df_grouped, min_trades=5):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"  {'Bucket':<25} {'Trades':>7} {'WR%':>6} {'PF':>6}  {'Signal'}")
    print(f"  {'─'*53}")
    for label, grp in df_grouped:
        if len(grp) < min_trades:
            continue
        p = pf(grp)
        w = wr(grp)
        net = grp["pnl"].sum()
        flag = " ✅ EDGE" if p >= 1.4 else (" ⚠ WEAK" if p >= 1.1 else " ❌ LOSS")
        print(f"  {str(label):<25} {len(grp):>7} {w:>6.1f} {p:>6.2f} {flag}  (${net:+.0f})")

# ──────────────────────────────────────────────────────
# 3. PF by Hour of Day
# ──────────────────────────────────────────────────────
show_table(
    "PF by Hour of Day (UTC)",
    tdf.groupby("hour"),
    min_trades=3
)

# ──────────────────────────────────────────────────────
# 4. PF by Day of Week
# ──────────────────────────────────────────────────────
dow_labels = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}
tdf["dow_label"] = tdf["dow"].map(dow_labels)
show_table(
    "PF by Day of Week",
    tdf.groupby("dow_label"),
    min_trades=3
)

# ──────────────────────────────────────────────────────
# 5. PF by Entry Probability Score Bucket
# ──────────────────────────────────────────────────────
tdf["prob_bucket"] = pd.cut(tdf["probability"], bins=[45,55,60,65,70,75,80,100],
                             labels=["45-55","55-60","60-65","65-70","70-75","75-80","80+"])
show_table(
    "PF by Probability Score at Entry",
    tdf.groupby("prob_bucket", observed=True),
    min_trades=3
)

# ──────────────────────────────────────────────────────
# 6. PF by ADX bucket at entry
# ──────────────────────────────────────────────────────
tdf["adx_bucket"] = pd.cut(tdf["adx"], bins=[0,15,20,25,30,100],
                            labels=["0-15","15-20","20-25","25-30","30+"])
show_table(
    "PF by ADX at Entry",
    tdf.groupby("adx_bucket", observed=True),
    min_trades=3
)

# ──────────────────────────────────────────────────────
# 7. PF by ATR Percentile at entry
# ──────────────────────────────────────────────────────
tdf["atr_bucket"] = pd.cut(tdf["atr_pct"], bins=[0, 0.25, 0.5, 0.66, 0.80, 1.01],
                            labels=["0-25%","25-50%","50-66%","66-80%","80-100%"])
show_table(
    "PF by ATR Percentile at Entry",
    tdf.groupby("atr_bucket", observed=True),
    min_trades=3
)

# ──────────────────────────────────────────────────────
# 8. PF by RSI at entry
# ──────────────────────────────────────────────────────
tdf["rsi_bucket"] = pd.cut(tdf["rsi"], bins=[0,30,40,50,60,70,100],
                            labels=["<30","30-40","40-50","50-60","60-70",">70"])
show_table(
    "PF by RSI at Entry",
    tdf.groupby("rsi_bucket", observed=True),
    min_trades=3
)

# ──────────────────────────────────────────────────────
# 9. PF by BB Position at entry
# ──────────────────────────────────────────────────────
tdf["bb_bucket"] = pd.cut(tdf["bb_pct"], bins=[-1, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0],
                           labels=["<0.1","0.1-0.2","0.2-0.3","0.3-0.5","0.5-1.0",">1.0"])
show_table(
    "PF by Bollinger Band Position at Entry",
    tdf.groupby("bb_bucket", observed=True),
    min_trades=3
)

# ──────────────────────────────────────────────────────
# 10. PF by Volume Ratio at entry
# ──────────────────────────────────────────────────────
tdf["vol_bucket"] = pd.cut(tdf["volume_ratio"], bins=[0, 0.5, 0.8, 1.0, 1.5, 2.0, 10.0],
                            labels=["<0.5","0.5-0.8","0.8-1.0","1.0-1.5","1.5-2.0",">2.0"])
show_table(
    "PF by Volume Ratio at Entry",
    tdf.groupby("vol_bucket", observed=True),
    min_trades=3
)

# ──────────────────────────────────────────────────────
# 11. Composite: Best combination — find what ACTUALLY works
# ──────────────────────────────────────────────────────
print(f"\n\n{'='*60}")
print("  COMPOSITE FILTER SEARCH — Where is PF > 1.4?")
print(f"{'='*60}")

best_combos = []
for adx_max in [15, 20, 25]:
    for hour_range in [(0,8),(8,16),(16,24)]:
        sub = tdf[(tdf["adx"] <= adx_max) &
                  (tdf["hour"] >= hour_range[0]) &
                  (tdf["hour"] < hour_range[1])]
        if len(sub) < 8:
            continue
        p = pf(sub)
        w = wr(sub)
        net = sub["pnl"].sum()
        best_combos.append((p, len(sub), w, f"ADX≤{adx_max} + Hour {hour_range[0]}-{hour_range[1]}UTC", net))

best_combos.sort(reverse=True)
print(f"\n  {'Combo':<35} {'Trades':>7} {'WR%':>6} {'PF':>6}  {'Net P&L'}")
print(f"  {'─'*65}")
for p_val, n, w, label, net in best_combos[:10]:
    flag = " ✅" if p_val >= 1.4 else " ⚠" if p_val >= 1.1 else " ❌"
    print(f"  {label:<35} {n:>7} {w:>6.1f} {p_val:>6.2f}{flag}  ${net:+.0f}")

print(f"\n{'='*60}")
print("  VERDICT")
print(f"{'='*60}")
overall_pf = pf(tdf)
overall_wr = wr(tdf)
print(f"\n  Overall OOS PF:  {overall_pf}")
print(f"  Overall OOS WR:  {overall_wr}%")
print(f"  Total Trades:    {len(tdf)}")
print(f"  Net P&L:         ${tdf['pnl'].sum():+.2f}")
