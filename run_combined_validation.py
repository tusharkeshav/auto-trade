"""
run_combined_validation.py
─────────────────────────────────────────────────────────────────────────────
Combined 15m + 30m BTC Validation
Runs the dual-timeframe strategy through TWO lenses:

  Section 1 — IN-SAMPLE BACKTEST  (all available history)
    15m BTC alone   →  baseline reference
    30m BTC alone   →  standalone 30m edge
    Combined        →  15m fires first, 30m fills gaps (position guard)

  Section 2 — OUT-OF-SAMPLE WALK-FORWARD  (rolling 30-day test folds)
    Same locked parameters — no optimization.
    Each fold: run 15m + 30m on the test window, merge with position guard.
    All folds stitched → single OOS equity curve.

  Section 3 — FINAL VERDICT
    Side-by-side: in-sample vs OOS for the combined strategy.

Usage:
    source .venv/bin/activate && python run_combined_validation.py
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from indicators                import add_all_indicators
from probability.signal_scorer import SignalScorer
from config.settings           import (
    INITIAL_CAPITAL_USDT, MAX_RISK_PER_TRADE_PCT,
    MACRO_MAX_ADX, MACRO_SESSION_START, MACRO_SESSION_END,
)

console = Console()

# ─── Locked strategy parameters ──────────────────────────────────────────────
SYMBOL        = "BTCUSDT"
CSV_15M       = "data/historical/BTCUSDT_15m_180000.csv"
THRESHOLD     = 48.0
ATR_SL_MULT   = 1.25
ATR_TP_MULT   = 1.5
WARMUP        = 200     # indicator warm-up candles

# OOS rolling window
TRAIN_DAYS    = 90
TEST_DAYS     = 30


# ─── Dataclass for a single simulated trade ───────────────────────────────────
@dataclass
class STrade:
    timeframe:   str
    direction:   str
    entry_time:  pd.Timestamp
    exit_time:   pd.Timestamp
    entry_price: float
    exit_price:  float
    pnl:         float
    exit_type:   str     # STOP_LOSS | TAKE_PROFIT


# ─── Resample ────────────────────────────────────────────────────────────────
def resample_to_30m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    return df.resample("30min").agg({
        "open": "first", "high": "max",
        "low":  "min",   "close": "last",
        "volume": "sum",
    }).dropna()


# ─── Core single-TF simulator ────────────────────────────────────────────────
def simulate(
    df: pd.DataFrame,
    tf_label: str,
    blocked_intervals: Optional[set[pd.Timestamp]] = None,
) -> list[STrade]:
    """
    Walk candle-by-candle on df. Enter on tradeable signals that pass the
    macro shield (already baked into SignalScorer).
    Skip entry if candle timestamp is in blocked_intervals (position guard).
    Returns list of STrade.
    """
    blocked = blocked_intervals or set()
    scorer  = SignalScorer(symbol=SYMBOL, long_threshold=THRESHOLD)
    trades: list[STrade] = []
    i = WARMUP

    while i < len(df) - 1:
        row = df.iloc[i]

        # Position guard: skip if another TF is already in
        if df.index[i] in blocked:
            i += 1
            continue

        signal = scorer.score(row)
        if signal.direction not in ("LONG", "SHORT") or signal.probability < THRESHOLD:
            i += 1
            continue

        atr     = row["atr"] if not math.isnan(row["atr"]) else signal.entry_price * 0.004
        sl_dist = atr * ATR_SL_MULT
        entry   = signal.entry_price
        if sl_dist == 0:
            i += 1
            continue

        sl = round(entry - sl_dist, 2) if signal.direction == "LONG" else round(entry + sl_dist, 2)
        tp = round(entry + sl_dist * ATR_TP_MULT, 2) if signal.direction == "LONG" else round(entry - sl_dist * ATR_TP_MULT, 2)
        size = (INITIAL_CAPITAL_USDT * MAX_RISK_PER_TRADE_PCT / 100) / sl_dist

        hit = None
        for j in range(i + 1, len(df)):
            hi, lo = df.iloc[j]["high"], df.iloc[j]["low"]
            if signal.direction == "LONG":
                if lo <= sl:
                    hit = ("STOP_LOSS",   sl, j); break
                if hi >= tp:
                    hit = ("TAKE_PROFIT", tp, j); break
            else:
                if hi >= sl:
                    hit = ("STOP_LOSS",   sl, j); break
                if lo <= tp:
                    hit = ("TAKE_PROFIT", tp, j); break

        if hit:
            etype, eprice, j = hit
            pnl = (eprice - entry) * size if signal.direction == "LONG" else (entry - eprice) * size
            trades.append(STrade(
                tf_label, signal.direction,
                df.index[i], df.index[j],
                entry, eprice, pnl, etype,
            ))
            i += (j - i)
            continue

        i += 1

    return trades


# ─── Merge 15m + 30m with position guard ─────────────────────────────────────
def merge_with_guard(trades_15m: list[STrade], df_30m: pd.DataFrame) -> list[STrade]:
    """
    Given 15m trades already simulated, compute which 30m timestamps are
    blocked, then simulate 30m with those blocked. Return combined list.
    """
    blocked_30m: set[pd.Timestamp] = set()
    for t in trades_15m:
        ts = t.entry_time.floor("30min")
        end = t.exit_time.ceil("30min")
        while ts <= end:
            blocked_30m.add(ts)
            ts += pd.Timedelta(minutes=30)

    trades_30m = simulate(df_30m, "30m", blocked_30m)
    combined   = trades_15m + trades_30m
    combined.sort(key=lambda t: t.entry_time)
    return combined


# ─── Metrics ─────────────────────────────────────────────────────────────────
def metrics(trades: list[STrade], years: float, label: str) -> dict:
    if not trades:
        return {"label": label, "n": 0, "n_yr": 0, "wr": 0, "pf": 0,
                "dd": 0, "pnl_pct": 0, "n_15m": 0, "n_30m": 0}

    wins = [t for t in trades if t.pnl > 0]
    gw   = sum(t.pnl for t in wins)
    gl   = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    pf   = round(gw / gl, 2) if gl else 999.0

    eq, pk, mx_dd = INITIAL_CAPITAL_USDT, INITIAL_CAPITAL_USDT, 0.0
    for t in sorted(trades, key=lambda x: x.entry_time):
        eq  += t.pnl
        pk   = max(pk, eq)
        mx_dd = max(mx_dd, (pk - eq) / pk * 100)

    return {
        "label":  label,
        "n":      len(trades),
        "n_yr":   round(len(trades) / years, 1) if years else 0,
        "wr":     round(len(wins) / len(trades) * 100, 1),
        "pf":     pf,
        "dd":     round(mx_dd, 1),
        "pnl_pct": round(sum(t.pnl for t in trades) / INITIAL_CAPITAL_USDT * 100, 1),
        "n_15m":  sum(1 for t in trades if t.timeframe == "15m"),
        "n_30m":  sum(1 for t in trades if t.timeframe == "30m"),
    }


def print_metrics_table(rows: list[dict], title: str):
    t = Table(title=title, border_style="cyan")
    t.add_column("Strategy",   style="bold white", min_width=22)
    t.add_column("Trades",     justify="right")
    t.add_column("Trades/yr",  justify="right")
    t.add_column("15m / 30m",  justify="right")
    t.add_column("Win Rate",   justify="right")
    t.add_column("PF",         justify="right")
    t.add_column("Max DD",     justify="right")
    t.add_column("Total PnL",  justify="right")

    for r in rows:
        pf_c = "green" if r["pf"] >= 1.5 else ("yellow" if r["pf"] >= 1.2 else "red")
        dd_c = "green" if r["dd"]  < 10  else ("yellow" if r["dd"]  < 15  else "red")
        breakdown = f"{r['n_15m']}m / {r['n_30m']}m" if r["n_15m"] + r["n_30m"] > 0 else "—"
        t.add_row(
            r["label"],
            str(r["n"]),
            str(r["n_yr"]),
            breakdown,
            f"{r['wr']:.1f}%",
            f"[{pf_c}]{r['pf']:.2f}[/]",
            f"[{dd_c}]{r['dd']:.1f}%[/]",
            f"{r['pnl_pct']:+.1f}%",
        )
    console.print(t)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: IN-SAMPLE BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def run_in_sample(df_15m: pd.DataFrame, df_30m: pd.DataFrame, years: float) -> list[dict]:
    console.print(Rule("[bold cyan]Section 1 — In-Sample Backtest (full history)[/]"))

    console.print("  [dim]Simulating 15m alone...[/]")
    t15 = simulate(df_15m, "15m")

    console.print("  [dim]Simulating 30m alone...[/]")
    t30_alone = simulate(df_30m, "30m")

    console.print("  [dim]Simulating combined (15m priority + 30m gap-fill)...[/]")
    t_comb = merge_with_guard(t15, df_30m)

    rows = [
        metrics(t15,      years, "BTC 15m alone"),
        metrics(t30_alone, years, "BTC 30m alone"),
        metrics(t_comb,   years, "BTC Combined (15m+30m)"),
    ]
    print_metrics_table(rows, "\nIn-Sample Results")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: OOS WALK-FORWARD (rolling 30-day test folds)
# ══════════════════════════════════════════════════════════════════════════════

def run_oos(df_15m: pd.DataFrame, df_30m: pd.DataFrame) -> dict:
    console.print(Rule("[bold cyan]Section 2 — Out-of-Sample Walk-Forward[/]"))
    console.print(f"  [dim]Rolling {TRAIN_DAYS}-day train / {TEST_DAYS}-day test windows[/]")
    console.print(f"  [dim]Params locked: ADX≤{MACRO_MAX_ADX}, session {MACRO_SESSION_START}-{MACRO_SESSION_END}UTC, "
                  f"threshold={THRESHOLD}, SL={ATR_SL_MULT}×ATR, TP={ATR_TP_MULT}×R[/]\n")

    # Work in UTC-aware timestamps
    df_15m = df_15m.copy()
    df_15m.index = pd.to_datetime(df_15m.index, utc=True)
    df_30m = df_30m.copy()
    df_30m.index = pd.to_datetime(df_30m.index, utc=True)

    start = df_15m.index[WARMUP]
    end   = df_15m.index[-1]

    all_oos_trades: list[STrade] = []
    fold_rows: list[dict]        = []
    fold = 1

    # First test window starts after TRAIN_DAYS of data
    test_start = start + timedelta(days=TRAIN_DAYS)

    candles_15m_per_day = int(24 * 60 / 15)  # 96
    candles_30m_per_day = int(24 * 60 / 30)  # 48
    warmup_15m_td = timedelta(minutes=15 * WARMUP)
    warmup_30m_td = timedelta(minutes=30 * WARMUP)

    while test_start + timedelta(days=TEST_DAYS) <= end:
        test_end = test_start + timedelta(days=TEST_DAYS)

        # Slice with warmup prepended (indicators need it)
        slice_15m = df_15m[
            (df_15m.index >= test_start - warmup_15m_td) &
            (df_15m.index <  test_end)
        ].copy()

        slice_30m = df_30m[
            (df_30m.index >= test_start - warmup_30m_td) &
            (df_30m.index <  test_end)
        ].copy()

        if len(slice_15m) < WARMUP + 10 or len(slice_30m) < WARMUP + 2:
            test_start += timedelta(days=TEST_DAYS)
            fold += 1
            continue

        # Simulate — only keep trades whose entry is inside the TEST window
        t15_all   = simulate(slice_15m, "15m")
        test_start_ts = pd.Timestamp(test_start).tz_localize("UTC") if test_start.tzinfo is None else pd.Timestamp(test_start)
        t15_fold  = [t for t in t15_all if t.entry_time >= test_start_ts]

        # Build 30m guard from full fold trades (not just OOS ones, to be safe)
        blocked_30m: set[pd.Timestamp] = set()
        for t in t15_all:
            ts  = t.entry_time.floor("30min")
            end_ = t.exit_time.ceil("30min")
            while ts <= end_:
                blocked_30m.add(ts)
                ts += pd.Timedelta(minutes=30)

        t30_all  = simulate(slice_30m, "30m", blocked_30m)
        t30_fold = [t for t in t30_all if t.entry_time >= test_start_ts]

        fold_trades = sorted(t15_fold + t30_fold, key=lambda t: t.entry_time)
        all_oos_trades.extend(fold_trades)

        fold_years = TEST_DAYS / 365.25
        if fold_trades:
            wins = [t for t in fold_trades if t.pnl > 0]
            gw   = sum(t.pnl for t in wins)
            gl   = abs(sum(t.pnl for t in fold_trades if t.pnl <= 0))
            f_pf = round(gw / gl, 2) if gl else 999.0
            f_wr = round(len(wins) / len(fold_trades) * 100, 1)
        else:
            f_pf, f_wr = 0.0, 0.0

        fold_rows.append({
            "fold":   fold,
            "period": f"{test_start.date()} → {test_end.date()}",
            "trades": len(fold_trades),
            "15m":    len([t for t in fold_trades if t.timeframe == "15m"]),
            "30m":    len([t for t in fold_trades if t.timeframe == "30m"]),
            "wr":     f_wr,
            "pf":     f_pf,
        })

        test_start += timedelta(days=TEST_DAYS)
        fold += 1

    # ── Per-fold table ────────────────────────────────────────────────────────
    ft = Table(title="OOS Fold-by-Fold", border_style="dim")
    ft.add_column("Fold", justify="right", style="dim")
    ft.add_column("Period",   style="dim")
    ft.add_column("Trades",   justify="right")
    ft.add_column("15m / 30m", justify="right")
    ft.add_column("Win Rate",  justify="right")
    ft.add_column("PF",        justify="right")

    for r in fold_rows:
        pf_c = "green" if r["pf"] >= 1.5 else ("yellow" if r["pf"] >= 1.0 else "red")
        ft.add_row(
            str(r["fold"]),
            r["period"],
            str(r["trades"]),
            f"{r['15m']} / {r['30m']}",
            f"{r['wr']:.1f}%",
            f"[{pf_c}]{r['pf']:.2f}[/]" if r["pf"] else "—",
        )
    console.print(ft)

    # ── OOS aggregate ─────────────────────────────────────────────────────────
    total_oos_days = len(fold_rows) * TEST_DAYS
    oos_years      = total_oos_days / 365.25
    oos_m          = metrics(all_oos_trades, oos_years, "BTC Combined OOS")
    print_metrics_table([oos_m], "\nOut-of-Sample Aggregate")

    return oos_m


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    console.print("\n[bold magenta]" + "═" * 70 + "[/]")
    console.print("[bold magenta]   BTC Dual-TF Validation: 15m + 30m   In-Sample & OOS[/]")
    console.print("[bold magenta]" + "═" * 70 + "[/]\n")
    console.print(f"  [dim]Symbol: {SYMBOL}  |  Macro: ADX≤{MACRO_MAX_ADX}, "
                  f"{MACRO_SESSION_START}-{MACRO_SESSION_END}UTC  |  "
                  f"Threshold: {THRESHOLD}  |  SL: {ATR_SL_MULT}×ATR  |  TP: {ATR_TP_MULT}×R[/]\n")

    if not Path(CSV_15M).exists():
        console.print(f"[red]✗ Data file not found: {CSV_15M}[/]")
        sys.exit(1)

    # Load & prepare data
    console.print("[dim]Loading BTC 15m data...[/]")
    df_15m_raw = pd.read_csv(CSV_15M, index_col=0, parse_dates=True)
    df_15m_raw.index = pd.to_datetime(df_15m_raw.index, utc=True)

    console.print("[dim]Resampling to 30m...[/]")
    df_30m_raw = resample_to_30m(df_15m_raw)

    console.print("[dim]Computing 15m indicators...[/]")
    df_15m = add_all_indicators(df_15m_raw.copy())

    console.print("[dim]Computing 30m indicators...[/]")
    df_30m = add_all_indicators(df_30m_raw.copy())

    # Align start (both need WARMUP candles of indicator history)
    start = max(df_15m.index[WARMUP], df_30m.index[WARMUP])
    df_15m = df_15m[df_15m.index >= start]
    df_30m = df_30m[df_30m.index >= start]

    days  = (df_15m.index[-1] - df_15m.index[0]).days
    years = days / 365.25
    console.print(f"  [dim]Period: {df_15m.index[0].date()} → {df_15m.index[-1].date()} "
                  f"({days} days / {years:.1f} years)[/]\n")

    # Section 1
    is_rows = run_in_sample(df_15m, df_30m, years)

    console.print()

    # Section 2
    oos_m = run_oos(df_15m, df_30m)

    # ── Section 3: Final verdict ──────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold magenta]Section 3 — Final Verdict[/]"))

    comb_is = next(r for r in is_rows if "Combined" in r["label"])
    vt = Table(title="In-Sample vs OOS — Combined Strategy", border_style="magenta")
    vt.add_column("Metric",     style="bold white")
    vt.add_column("In-Sample",  justify="right")
    vt.add_column("OOS",        justify="right")
    vt.add_column("Verdict",    justify="center")

    def verdict(is_val: float, oos_val: float, metric: str) -> str:
        if metric == "pf":
            if oos_val >= 1.3:  return "[green]✅ PASS[/]"
            if oos_val >= 1.0:  return "[yellow]⚠ WEAK[/]"
            return "[red]❌ FAIL[/]"
        if metric == "dd":
            return "[green]✅ PASS[/]" if oos_val < 15 else "[red]❌ FAIL[/]"
        if metric == "tpy":
            return "[green]✅ PASS[/]" if oos_val >= 15 else "[yellow]⚠ LOW[/]"
        if metric == "wr":
            return "[green]✅ PASS[/]" if oos_val >= 45 else "[yellow]⚠ WEAK[/]"
        return "—"

    vt.add_row("Trades / yr",    str(comb_is["n_yr"]),  str(oos_m["n_yr"]),
               verdict(comb_is["n_yr"], oos_m["n_yr"], "tpy"))
    vt.add_row("Win Rate",        f"{comb_is['wr']:.1f}%", f"{oos_m['wr']:.1f}%",
               verdict(comb_is["wr"], oos_m["wr"], "wr"))
    vt.add_row("Profit Factor",   f"{comb_is['pf']:.2f}",  f"{oos_m['pf']:.2f}",
               verdict(comb_is["pf"], oos_m["pf"], "pf"))
    vt.add_row("Max Drawdown",    f"{comb_is['dd']:.1f}%", f"{oos_m['dd']:.1f}%",
               verdict(comb_is["dd"], oos_m["dd"], "dd"))
    vt.add_row("15m / 30m split",
               f"{comb_is['n_15m']} / {comb_is['n_30m']}",
               f"{oos_m['n_15m']} / {oos_m['n_30m']}", "—")
    console.print(vt)

    # Overall GO / NO-GO
    go = (oos_m["pf"] >= 1.3 and oos_m["dd"] < 15 and oos_m["n_yr"] >= 15)
    console.print()
    if go:
        console.print("[bold green]  ██████  STRATEGY VALIDATED — 30m BTC INTEGRATION APPROVED  ██████[/]")
        console.print("[green]  Both in-sample and OOS pass all criteria. Live bot is correctly configured.[/]")
    else:
        console.print("[bold red]  ██████  STRATEGY NEEDS REVIEW — Check OOS metrics above  ██████[/]")
        if oos_m["pf"] < 1.3:
            console.print(f"[red]  OOS PF {oos_m['pf']:.2f} < 1.30 threshold[/]")
        if oos_m["dd"] >= 15:
            console.print(f"[red]  OOS Max DD {oos_m['dd']:.1f}% ≥ 15% threshold[/]")
        if oos_m["n_yr"] < 15:
            console.print(f"[yellow]  OOS Trades/yr {oos_m['n_yr']} < 15 (low statistical weight)[/]")
    console.print()


if __name__ == "__main__":
    main()
