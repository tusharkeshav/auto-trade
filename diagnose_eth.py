"""
diagnose_eth.py
─────────────────────────────────────────────────────────────────────────────
ETH Diagnostic — 3 steps:
  Step 1: ETH 15m OOS fold-by-fold (where is PF dragged down?)
  Step 2: ETH 15m condition sensitivity (ADX / hour / RSI buckets)
  Step 3: ETH 30m vs 15m signal conditions comparison

Usage: source .venv/bin/activate && python diagnose_eth.py
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import math, sys, warnings
from datetime import timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.rule import Rule

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from indicators import add_all_indicators
from probability.signal_scorer import SignalScorer
from config.settings import INITIAL_CAPITAL_USDT, MAX_RISK_PER_TRADE_PCT

console = Console()

SYMBOL      = "ETHUSDT"
CSV_15M     = "data/historical/ETHUSDT_15m_87000.csv"
THRESHOLD   = 48.0
ATR_SL_MULT = 1.25
ATR_TP_MULT = 1.5
WARMUP      = 200
TEST_DAYS   = 30
TRAIN_DAYS  = 90


# ─── Resample ────────────────────────────────────────────────────────────────

def resample_30m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    return df.resample("30min").agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna()


# ─── Simulator — returns rich trade dicts with entry-condition context ────────

def simulate_rich(df: pd.DataFrame, tf: str) -> list[dict]:
    """Like simulate() but includes ADX, RSI, hour, BB at entry for analysis."""
    scorer = SignalScorer(symbol=SYMBOL, long_threshold=THRESHOLD)
    trades = []
    i = WARMUP

    while i < len(df) - 1:
        row = df.iloc[i]
        sig = scorer.score(row)

        if sig.direction not in ("LONG","SHORT") or sig.probability < THRESHOLD:
            i += 1; continue

        atr     = row["atr"] if not math.isnan(row.get("atr", float("nan"))) else sig.entry_price * 0.004
        sl_dist = atr * ATR_SL_MULT
        entry   = sig.entry_price
        if sl_dist == 0: i += 1; continue

        sl   = round(entry - sl_dist, 2) if sig.direction == "LONG" else round(entry + sl_dist, 2)
        tp   = round(entry + sl_dist * ATR_TP_MULT, 2) if sig.direction == "LONG" else round(entry - sl_dist * ATR_TP_MULT, 2)
        size = (INITIAL_CAPITAL_USDT * MAX_RISK_PER_TRADE_PCT / 100) / sl_dist

        hit = None
        for j in range(i + 1, len(df)):
            hi, lo = df.iloc[j]["high"], df.iloc[j]["low"]
            if sig.direction == "LONG":
                if lo <= sl: hit = ("SL", sl, j); break
                if hi >= tp: hit = ("TP", tp, j); break
            else:
                if hi >= sl: hit = ("SL", sl, j); break
                if lo <= tp: hit = ("TP", tp, j); break

        if hit:
            etype, eprice, j = hit
            pnl = (eprice - entry)*size if sig.direction == "LONG" else (entry - eprice)*size
            trades.append({
                "tf":        tf,
                "direction": sig.direction,
                "entry_time": df.index[i],
                "exit_time":  df.index[j],
                "pnl":        pnl,
                "exit_type":  etype,
                "prob":       sig.probability,
                "adx":        row.get("adx", np.nan),
                "rsi":        row.get("rsi", np.nan),
                "bb_pct":     row.get("bb_pct", np.nan),
                "vol_ratio":  row.get("volume_ratio", np.nan),
                "hour":       df.index[i].hour,
                "dow":        df.index[i].dayofweek,
            })
            i += (j - i)
            continue
        i += 1

    return trades


# ─── Metrics helpers ─────────────────────────────────────────────────────────

def pf(trades):
    if not trades: return 0.0
    gw = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    return round(gw / gl, 2) if gl else 999.0

def wr(trades):
    if not trades: return 0.0
    return round(sum(1 for t in trades if t["pnl"] > 0) / len(trades) * 100, 1)

def show_bucket_table(title: str, groups: list[tuple[str, list[dict]]], min_trades: int = 3):
    t = Table(title=title, border_style="dim")
    t.add_column("Bucket",  style="dim", min_width=16)
    t.add_column("Trades",  justify="right")
    t.add_column("Win Rate", justify="right")
    t.add_column("PF",       justify="right")
    t.add_column("Net PnL",  justify="right")
    t.add_column("Signal",   justify="left")

    for label, subset in groups:
        if len(subset) < min_trades: continue
        p  = pf(subset)
        w  = wr(subset)
        net = sum(t["pnl"] for t in subset)
        pf_c = "green" if p >= 1.5 else ("yellow" if p >= 1.0 else "red")
        flag  = "✅ EDGE" if p >= 1.4 else ("⚠ WEAK" if p >= 1.0 else "❌ LOSS")
        t.add_row(str(label), str(len(subset)), f"{w:.1f}%",
                  f"[{pf_c}]{p:.2f}[/]", f"${net:+.0f}", flag)
    console.print(t)


# ════════════════════════════════════════════════════════════════════
#  STEP 1 — ETH 15m OOS fold-by-fold
# ════════════════════════════════════════════════════════════════════

def step1_oos_folds(df_15m: pd.DataFrame):
    console.print(Rule("[bold cyan]Step 1 — ETH 15m OOS Fold-by-Fold[/]"))
    console.print("  [dim]Identifies which periods drag PF below 1.0[/]\n")

    df_15m = df_15m.copy()
    df_15m.index = pd.to_datetime(df_15m.index, utc=True)

    warmup_td = timedelta(minutes=15 * WARMUP)
    start     = df_15m.index[WARMUP]
    end       = df_15m.index[-1]
    test_start = start + timedelta(days=TRAIN_DAYS)

    ft = Table(title="ETH 15m OOS Folds", border_style="dim")
    ft.add_column("Fold",   justify="right", style="dim")
    ft.add_column("Period", style="dim")
    ft.add_column("Trades", justify="right")
    ft.add_column("WR%",    justify="right")
    ft.add_column("PF",     justify="right")
    ft.add_column("Net",    justify="right")

    all_oos = []
    fold = 1
    while test_start + timedelta(days=TEST_DAYS) <= end:
        test_end = test_start + timedelta(days=TEST_DAYS)
        slc = df_15m[(df_15m.index >= test_start - warmup_td) & (df_15m.index < test_end)].copy()
        if len(slc) < WARMUP + 5:
            test_start += timedelta(days=TEST_DAYS); fold += 1; continue

        all_t = simulate_rich(slc, "15m")
        ts    = pd.Timestamp(test_start) if test_start.tzinfo else pd.Timestamp(test_start, tz="UTC")
        fold_t = [t for t in all_t if t["entry_time"] >= ts]
        all_oos.extend(fold_t)

        f_pf  = pf(fold_t)
        f_wr  = wr(fold_t)
        net   = sum(t["pnl"] for t in fold_t)
        pf_c  = "green" if f_pf >= 1.3 else ("yellow" if f_pf >= 1.0 else "red")

        ft.add_row(
            str(fold),
            f"{test_start.date()} → {test_end.date()}",
            str(len(fold_t)),
            f"{f_wr:.1f}%",
            f"[{pf_c}]{f_pf:.2f}[/]" if fold_t else "—",
            f"${net:+.0f}",
        )
        test_start += timedelta(days=TEST_DAYS)
        fold += 1

    console.print(ft)
    oos_years = len([r for r in range(fold)]) * TEST_DAYS / 365.25
    console.print(f"\n  OOS aggregate: {len(all_oos)} trades  PF={pf(all_oos):.2f}  WR={wr(all_oos):.1f}%\n")
    return all_oos


# ════════════════════════════════════════════════════════════════════
#  STEP 2 — ETH 15m condition sensitivity
# ════════════════════════════════════════════════════════════════════

def step2_conditions(trades_15m: list[dict]):
    console.print(Rule("[bold cyan]Step 2 — ETH 15m Entry Condition Sensitivity[/]"))

    # ADX buckets
    def adx_bucket(t):
        adx = t["adx"]
        if np.isnan(adx): return "unknown"
        if adx <= 15: return "0-15"
        if adx <= 20: return "15-20"
        if adx <= 25: return "20-25"
        return "25+"

    show_bucket_table("PF by ADX at Entry", [
        (k, [t for t in trades_15m if adx_bucket(t) == k])
        for k in ["0-15","15-20","20-25","25+"]
    ], min_trades=2)

    # Hour buckets
    show_bucket_table("PF by Hour (UTC)", [
        (f"{h:02d}:00", [t for t in trades_15m if t["hour"] == h])
        for h in range(16, 24)
    ], min_trades=2)

    # Day of week
    dow_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    show_bucket_table("PF by Day of Week", [
        (dow_map[d], [t for t in trades_15m if t["dow"] == d])
        for d in range(7)
    ], min_trades=2)

    # RSI buckets
    def rsi_bucket(t):
        r = t["rsi"]
        if np.isnan(r): return "unknown"
        if r < 35: return "<35 (oversold)"
        if r < 45: return "35-45"
        if r < 55: return "45-55"
        if r < 65: return "55-65"
        return ">65 (overbought)"

    show_bucket_table("PF by RSI at Entry", [
        (k, [t for t in trades_15m if rsi_bucket(t) == k])
        for k in ["<35 (oversold)","35-45","45-55","55-65",">65 (overbought)"]
    ], min_trades=2)

    # Composite: tighter ADX
    console.print("\n[bold]Composite test: What if ADX ≤ 20 for ETH?[/]")
    adx20 = [t for t in trades_15m if not np.isnan(t["adx"]) and t["adx"] <= 20]
    adx25 = [t for t in trades_15m if not np.isnan(t["adx"]) and t["adx"] <= 25]
    adx25_only = [t for t in adx25 if not np.isnan(t["adx"]) and t["adx"] > 20]

    show_bucket_table("ADX Tightening Composite", [
        ("ADX ≤ 20",          adx20),
        ("ADX 20–25 (marginal)", adx25_only),
        ("ADX ≤ 25 (current)", adx25),
    ], min_trades=2)


# ════════════════════════════════════════════════════════════════════
#  STEP 3 — ETH 30m vs 15m signal comparison
# ════════════════════════════════════════════════════════════════════

def step3_30m_comparison(trades_15m: list[dict], trades_30m: list[dict]):
    console.print(Rule("[bold cyan]Step 3 — ETH 30m vs 15m Signal Conditions[/]"))
    console.print("  [dim]Why does 30m (PF ~1.5) outperform 15m (PF ~1.07)?[/]\n")

    def avg(trades, field):
        vals = [t[field] for t in trades if not np.isnan(t.get(field, float("nan")))]
        return round(np.mean(vals), 1) if vals else float("nan")

    ct = Table(title="Signal Context: 15m vs 30m", border_style="cyan")
    ct.add_column("Metric",     style="bold white")
    ct.add_column("15m",        justify="right")
    ct.add_column("30m",        justify="right")
    ct.add_column("Δ",          justify="right")

    metrics_list = [
        ("Total trades",    len(trades_15m),         len(trades_30m)),
        ("Win Rate",        f"{wr(trades_15m):.1f}%", f"{wr(trades_30m):.1f}%"),
        ("Profit Factor",   f"{pf(trades_15m):.2f}",  f"{pf(trades_30m):.2f}"),
        ("Avg ADX at entry", avg(trades_15m,"adx"),   avg(trades_30m,"adx")),
        ("Avg RSI at entry", avg(trades_15m,"rsi"),   avg(trades_30m,"rsi")),
        ("Avg Prob score",  avg(trades_15m,"prob"),   avg(trades_30m,"prob")),
    ]
    for label, v15, v30 in metrics_list:
        try:
            delta = f"{float(v30)-float(v15):+.1f}"
        except Exception:
            delta = "—"
        ct.add_row(str(label), str(v15), str(v30), delta)
    console.print(ct)

    # ADX distribution of 30m signals
    def adx_dist(trades):
        buckets = {"≤15":0,"15-20":0,"20-25":0,"25+":0}
        for t in trades:
            adx = t["adx"]
            if np.isnan(adx): continue
            if adx <= 15: buckets["≤15"] += 1
            elif adx <= 20: buckets["15-20"] += 1
            elif adx <= 25: buckets["20-25"] += 1
            else: buckets["25+"] += 1
        return buckets

    d15 = adx_dist(trades_15m)
    d30 = adx_dist(trades_30m)

    at = Table(title="ADX Distribution at Entry", border_style="dim")
    at.add_column("ADX Range", style="dim")
    at.add_column("15m trades", justify="right")
    at.add_column("30m trades", justify="right")
    for k in ["≤15","15-20","20-25","25+"]:
        at.add_row(k, str(d15.get(k,0)), str(d30.get(k,0)))
    console.print(at)

    console.print("\n[bold]Interpretation:[/]")
    console.print("  If 30m signals cluster at lower ADX → 30m naturally filters trending conditions better")
    console.print("  If 30m signals have higher avg prob → 30m scorer is more selective at the 30m granularity")


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    console.print("\n[bold magenta]" + "═"*60 + "[/]")
    console.print("[bold magenta]   ETH Diagnostic — 15m vs 30m Analysis[/]")
    console.print("[bold magenta]" + "═"*60 + "[/]\n")

    if not Path(CSV_15M).exists():
        console.print(f"[red]✗ {CSV_15M} not found[/]"); sys.exit(1)

    console.print("[dim]Loading ETH 15m data...[/]")
    df_raw = pd.read_csv(CSV_15M, index_col=0, parse_dates=True)
    df_raw.index = pd.to_datetime(df_raw.index, utc=True)

    console.print("[dim]Resampling to 30m...[/]")
    df_30m_raw = resample_30m(df_raw)

    console.print("[dim]Computing indicators...[/]")
    df_15m = add_all_indicators(df_raw.copy())
    df_30m = add_all_indicators(df_30m_raw.copy())

    start = max(df_15m.index[WARMUP], df_30m.index[WARMUP])
    df_15m = df_15m[df_15m.index >= start]
    df_30m = df_30m[df_30m.index >= start]

    days  = (df_15m.index[-1] - df_15m.index[0]).days
    years = days / 365.25
    console.print(f"  [dim]Period: {df_15m.index[0].date()} → {df_15m.index[-1].date()} "
                  f"({days}d / {years:.1f}yr)[/]\n")

    # Step 1
    oos_trades = step1_oos_folds(df_15m)

    # Step 2 — use full in-sample trades for richer sample
    console.print("[dim]Running in-sample 15m for condition analysis...[/]")
    trades_15m_full = simulate_rich(df_15m, "15m")
    console.print(f"  {len(trades_15m_full)} in-sample 15m trades\n")
    step2_conditions(trades_15m_full)

    # Step 3
    console.print("[dim]Running in-sample 30m...[/]")
    trades_30m_full = simulate_rich(df_30m, "30m")
    console.print(f"  {len(trades_30m_full)} in-sample 30m trades\n")
    step3_30m_comparison(trades_15m_full, trades_30m_full)

    console.print("\n[bold]Summary of findings:[/]")
    p15 = pf(trades_15m_full)
    p30 = pf(trades_30m_full)
    console.print(f"  ETH 15m PF: [{'green' if p15>=1.3 else 'red'}]{p15:.2f}[/]")
    console.print(f"  ETH 30m PF: [{'green' if p30>=1.3 else 'red'}]{p30:.2f}[/]")

    if p30 >= 1.3 and p15 < 1.2:
        console.print("\n  [bold yellow]⚠  RECOMMENDATION: Skip ETH 15m, add ETH 30m instead[/]")
        console.print("  [dim]30m timeframe appears to filter noise that hurts the 15m edge on ETH[/]")
    elif p15 >= 1.3:
        console.print("\n  [bold green]✅  ETH 15m edge confirmed — safe to add ETH 30m too[/]")
    else:
        console.print("\n  [bold red]❌  ETH edge is structurally weak on both timeframes[/]")
        console.print("  [dim]Consider excluding ETH entirely or tightening ADX filter[/]")
    console.print()


if __name__ == "__main__":
    main()
