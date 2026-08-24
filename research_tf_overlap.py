"""
research_tf_overlap.py
─────────────────────────────────────────────────────────────────────────────
Phase 1C — Option C Research: Signal Overlap Between 15m and 30m

The key research question: When the 30m scores a tradeable signal, how often
is there already a 15m position open (or a 15m signal within 30 minutes)?

If overlap is high → 30m mostly fires when 15m is already in → low additive value
If overlap is low  → 30m fires independently → genuine trade count boost

Methodology:
  1. Score every candle on 15m → mark tradeable signals (≥ threshold, pass macro)
  2. Score every candle on 30m → mark tradeable signals
  3. For each 30m signal: check if a 15m trade was entered within ±30 min
  4. Report: overlap %, additive signals, expected combined trade count

Usage:
    source .venv/bin/activate && python research_tf_overlap.py
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
import json
import warnings
from pathlib import Path
from datetime import timedelta

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from indicators                import add_all_indicators
from probability.signal_scorer import SignalScorer
from config.settings           import MACRO_MAX_ADX, MACRO_SESSION_START, MACRO_SESSION_END

console = Console()

THRESHOLD = 48.0
ASSETS = {
    "BTCUSDT": "data/historical/BTCUSDT_15m_180000.csv",
    "ETHUSDT": "data/historical/ETHUSDT_15m_87000.csv",
}


# ─── Resample ────────────────────────────────────────────────────────────────

def resample_to_30m(df_15m: pd.DataFrame) -> pd.DataFrame:
    df = df_15m.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    return df.resample("30min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()


# ─── Score all candles on a given dataframe ───────────────────────────────────

def find_tradeable_signals(df: pd.DataFrame, symbol: str, label: str) -> list[dict]:
    """
    Walk every candle (after warmup) and return a list of tradeable signal timestamps.
    A signal is tradeable if:
      - direction != NO_TRADE
      - probability >= THRESHOLD
      - (macro shield already enforced inside SignalScorer)
    """
    scorer  = SignalScorer(symbol=symbol, long_threshold=THRESHOLD)
    WARMUP  = 200
    signals = []

    for i in range(WARMUP, len(df) - 1):
        row = df.iloc[i]
        sig = scorer.score(row)
        if sig.direction in ("LONG", "SHORT") and sig.probability >= THRESHOLD:
            signals.append({
                "timestamp": df.index[i],
                "direction": sig.direction,
                "probability": sig.probability,
                "timeframe": label,
            })

    logger.info(f"{symbol} {label}: {len(signals)} tradeable signals in {len(df)-WARMUP:,} candles")
    return signals


# ─── Overlap analysis ─────────────────────────────────────────────────────────

def analyse_overlap(symbol: str, csv_path: str) -> dict:
    console.print(f"\n[bold cyan]▶  {symbol}  —  Overlap Analysis[/]")

    # Load 15m
    df_15m = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df_15m.index = pd.to_datetime(df_15m.index, utc=True)
    df_15m_ind = add_all_indicators(df_15m.copy())

    # Build 30m from same data
    df_30m = resample_to_30m(df_15m)
    df_30m_ind = add_all_indicators(df_30m.copy())

    # Restrict to the date range present in both
    start = max(df_15m_ind.index[200], df_30m_ind.index[200])
    df_15m_ind = df_15m_ind[df_15m_ind.index >= start]
    df_30m_ind = df_30m_ind[df_30m_ind.index >= start]

    # Score signals
    console.print("  [dim]Scoring 15m signals...[/]")
    sigs_15m = find_tradeable_signals(df_15m_ind, symbol, "15m")

    console.print("  [dim]Scoring 30m signals...[/]")
    sigs_30m = find_tradeable_signals(df_30m_ind, symbol, "30m")

    if not sigs_30m:
        return {"symbol": symbol, "sigs_15m": len(sigs_15m), "sigs_30m": 0,
                "overlap_count": 0, "additive_count": 0, "overlap_pct": 0}

    # Build set of 15m signal timestamps for fast lookup
    ts_15m = {s["timestamp"] for s in sigs_15m}

    # For each 30m signal, check if a 15m signal fired within ±30 minutes
    # "Within ±30 min" = the 30m candle covers the same 15m entry window
    WINDOW = timedelta(minutes=30)
    overlap_count  = 0
    additive_count = 0
    additive_sigs  = []

    for sig_30 in sigs_30m:
        t30 = sig_30["timestamp"]
        # Check if any 15m signal is within the 30m window
        nearby_15m = any(
            abs((t30 - t15).total_seconds()) <= WINDOW.total_seconds()
            for t15 in ts_15m
        )
        if nearby_15m:
            overlap_count += 1
        else:
            additive_count += 1
            additive_sigs.append(sig_30)

    overlap_pct = round(overlap_count / len(sigs_30m) * 100, 1)

    # Calculate period
    period_days = (df_15m_ind.index[-1] - df_15m_ind.index[0]).days
    years = period_days / 365.25

    return {
        "symbol":          symbol,
        "period_years":    round(years, 1),
        "sigs_15m":        len(sigs_15m),
        "sigs_30m":        len(sigs_30m),
        "overlap_count":   overlap_count,
        "additive_count":  additive_count,
        "overlap_pct":     overlap_pct,
        "additive_per_yr": round(additive_count / years, 1),
        "sigs_15m_per_yr": round(len(sigs_15m) / years, 1),
        "combined_per_yr": round((len(sigs_15m) + additive_count) / years, 1),
    }


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    console.print("\n[bold magenta]═══  Option C Research — Phase 1C: Signal Overlap Analysis  ═══[/]")
    console.print(f"[dim]Macro Shield: ADX ≤ {MACRO_MAX_ADX}  |  Session: {MACRO_SESSION_START}:00–{MACRO_SESSION_END}:00 UTC[/]")
    console.print(f"[dim]Overlap window: ±30 minutes (same session candle)[/]\n")

    results = []
    for symbol, csv_path in ASSETS.items():
        if not Path(csv_path).exists():
            console.print(f"[red]  ✗ {csv_path} not found — skipping[/]")
            continue
        r = analyse_overlap(symbol, csv_path)
        results.append(r)

    # ─── Print results ────────────────────────────────────────────────────────
    table = Table(title="\nSignal Overlap Results", border_style="cyan")
    table.add_column("Asset",            style="bold white")
    table.add_column("Period",           style="dim")
    table.add_column("15m Signals/yr",   justify="right")
    table.add_column("30m Signals",      justify="right")
    table.add_column("Overlap",          justify="right")
    table.add_column("Additive/yr",      justify="right")
    table.add_column("Combined/yr",      justify="right")

    for r in results:
        ov_color = "green" if r["overlap_pct"] < 60 else "red"
        table.add_row(
            r["symbol"],
            f"{r['period_years']}y",
            str(r["sigs_15m_per_yr"]),
            str(r["sigs_30m"]),
            f"[{ov_color}]{r['overlap_pct']:.1f}%[/]",
            f"[green]+{r['additive_per_yr']}[/]",
            f"[bold]{r['combined_per_yr']}[/]",
        )

    console.print(table)

    console.print("\n[bold]Interpretation:[/]")
    for r in results:
        console.print(
            f"  [cyan]{r['symbol']}:[/]  {r['sigs_15m_per_yr']}/yr (15m)  +  "
            f"[green]+{r['additive_per_yr']}/yr additive (30m)[/]  =  "
            f"[bold]{r['combined_per_yr']}/yr combined[/]"
        )

    total_15m = sum(r["sigs_15m_per_yr"] for r in results)
    total_add = sum(r["additive_per_yr"] for r in results)
    console.print(f"\n  [bold white]TOTAL across BTC+ETH:  "
                  f"{total_15m:.0f}/yr (15m)  +  {total_add:.0f}/yr (30m additive)  "
                  f"=  {total_15m + total_add:.0f}/yr combined[/]")

    console.print("\n[bold]Go/No-Go criteria for overlap:[/]")
    console.print("  Overlap < 60%  →  30m adds genuine independent signals")
    for r in results:
        go = r["overlap_pct"] < 60
        icon = "✅" if go else "❌"
        console.print(f"  {icon}  {r['symbol']}: overlap={r['overlap_pct']}%")

    # Save for summary
    out = Path("research_out")
    out.mkdir(exist_ok=True)
    (out / "overlap_analysis.json").write_text(json.dumps(results, indent=2))
    console.print(f"\n[dim]Results saved → research_out/overlap_analysis.json[/]")


if __name__ == "__main__":
    main()
