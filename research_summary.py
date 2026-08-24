"""
research_summary.py
─────────────────────────────────────────────────────────────────────────────
Phase 1 Final — Option C Research: Go / No-Go Decision

Reads JSON outputs from the three research scripts and prints a clean
decision table against the Phase 2 go/no-go criteria.

Run AFTER:
  1. python research_30m_standalone.py
  2. python research_tf_overlap.py
  3. python research_dual_tf_backtest.py

Usage:
    source .venv/bin/activate && python research_summary.py
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table   import Table

console = Console()

OUT_DIR = Path("research_out")


def load(filename: str) -> list | None:
    path = OUT_DIR / filename
    if not path.exists():
        console.print(f"[red]  ✗ {filename} not found — run the research scripts first[/]")
        return None
    return json.loads(path.read_text())


def check(value: float, threshold: float, direction: str = "above") -> str:
    """Return green tick or red cross based on comparison."""
    if direction == "above":
        return "✅" if value >= threshold else "❌"
    else:
        return "✅" if value <= threshold else "❌"


def main():
    console.print("\n[bold magenta]═══════════════════════════════════════════════════════════════════[/]")
    console.print("[bold magenta]   Option C Research Summary — Go / No-Go Decision for Phase 2    [/]")
    console.print("[bold magenta]═══════════════════════════════════════════════════════════════════[/]\n")

    standalone = load("standalone_30m.json")
    overlap    = load("overlap_analysis.json")
    dual_tf    = load("dual_tf_backtest.json")

    all_loaded = all(x is not None for x in [standalone, overlap, dual_tf])
    if not all_loaded:
        console.print("\n[yellow]Some results files are missing. Run all 3 research scripts first.[/]")
        return

    # ─── Section 1: 30m Standalone ───────────────────────────────────────────
    console.print("[bold cyan]1. 30m Standalone Edge[/]  (Phase 1B)")
    console.print("   Does the 30m timeframe have a genuine edge under our locked macro rules?\n")

    t1 = Table(border_style="dim")
    t1.add_column("Asset",    style="bold white")
    t1.add_column("Period",   style="dim")
    t1.add_column("Trades/yr", justify="right")
    t1.add_column("Win Rate", justify="right")
    t1.add_column("PF ≥ 1.20", justify="center")
    t1.add_column("WR ≥ 45%", justify="center")
    t1.add_column("DD < 15%", justify="center")
    t1.add_column("PASS?",    justify="center")

    for r in standalone:
        pf_ok = r["profit_factor"] >= 1.20
        wr_ok = r["win_rate"] >= 45
        dd_ok = r["max_dd_pct"] < 15
        passes = pf_ok and wr_ok and dd_ok
        t1.add_row(
            r["symbol"],
            f"{r['years']}y",
            str(r["trades_per_year"]),
            f"{r['win_rate']:.1f}%",
            f"{'✅' if pf_ok else '❌'} {r['profit_factor']:.2f}",
            f"{'✅' if wr_ok else '❌'} {r['win_rate']:.1f}%",
            f"{'✅' if dd_ok else '❌'} {r['max_dd_pct']:.1f}%",
            f"[bold {'green' if passes else 'red'}]{'GO' if passes else 'NO-GO'}[/]",
        )
    console.print(t1)

    # ─── Section 2: Overlap Analysis ─────────────────────────────────────────
    console.print("\n[bold cyan]2. Signal Independence (Overlap)[/]  (Phase 1C)")
    console.print("   How often does 30m fire when no 15m position is already open?\n")

    t2 = Table(border_style="dim")
    t2.add_column("Asset",         style="bold white")
    t2.add_column("15m / yr",      justify="right")
    t2.add_column("30m total",     justify="right")
    t2.add_column("Overlap",       justify="right")
    t2.add_column("Additive / yr", justify="right")
    t2.add_column("Combined / yr", justify="right")
    t2.add_column("Overlap < 60%", justify="center")

    for r in overlap:
        ov_ok = r["overlap_pct"] < 60
        t2.add_row(
            r["symbol"],
            str(r["sigs_15m_per_yr"]),
            str(r["sigs_30m"]),
            f"{'✅' if ov_ok else '❌'} {r['overlap_pct']:.1f}%",
            f"+{r['additive_per_yr']}",
            f"[bold]{r['combined_per_yr']}[/]",
            f"{'✅' if ov_ok else '❌'}",
        )
    console.print(t2)

    # ─── Section 3: Combined Backtest ─────────────────────────────────────────
    console.print("\n[bold cyan]3. Combined Portfolio (15m Priority + 30m Gap Fill)[/]  (Phase 1D)")
    console.print("   Does the combined system improve trade count without hurting PF or DD?\n")

    t3 = Table(border_style="dim")
    t3.add_column("Configuration", style="bold white", min_width=22)
    t3.add_column("Trades/yr",     justify="right")
    t3.add_column("PF",            justify="right")
    t3.add_column("Win Rate",      justify="right")
    t3.add_column("Max DD",        justify="right")
    t3.add_column("PF ≥ 1.30",     justify="center")
    t3.add_column("DD < 15%",      justify="center")
    t3.add_column("≥ 30/yr",       justify="center")

    for r in dual_tf:
        for key, m in [("15m alone", r["metrics_15m"]),
                       ("30m alone", r["metrics_30m"]),
                       ("Combined",  r["metrics_combined"])]:
            pf_ok    = m["pf"] >= 1.30
            dd_ok    = m["max_dd_pct"] < 15
            count_ok = m["trades_per_yr"] >= 30
            t3.add_row(
                f"{r['symbol']} {key}",
                str(m["trades_per_yr"]),
                f"{'✅' if pf_ok else '❌'} {m['pf']:.2f}",
                f"{m['win_rate']:.1f}%",
                f"{'✅' if dd_ok else '❌'} {m['max_dd_pct']:.1f}%",
                f"{'✅' if pf_ok else '❌'}",
                f"{'✅' if dd_ok else '❌'}",
                f"{'✅' if count_ok else '❌'}",
            )
        t3.add_section()

    console.print(t3)

    # ─── Final Verdict ────────────────────────────────────────────────────────
    console.print("\n[bold]═══  FINAL VERDICT  ═══[/]\n")

    # Aggregate checks
    standalone_passes = all(
        r["profit_factor"] >= 1.20 and r["win_rate"] >= 45 and r["max_dd_pct"] < 15
        for r in standalone
    )
    overlap_passes = all(r["overlap_pct"] < 60 for r in overlap)

    combined_passes = all(
        r["metrics_combined"]["pf"] >= 1.30
        and r["metrics_combined"]["max_dd_pct"] < 15
        and r["metrics_combined"]["trades_per_yr"] >= 30
        for r in dual_tf
    )

    # Combined trades/yr total across assets
    total_combined_per_yr = sum(r["metrics_combined"]["trades_per_yr"] for r in dual_tf)

    checks = [
        ("30m has standalone edge (PF ≥ 1.20, WR ≥ 45%, DD < 15%)", standalone_passes),
        ("Signals are mostly independent (Overlap < 60%)",             overlap_passes),
        ("Combined PF ≥ 1.30 across all assets",                      combined_passes),
    ]

    all_pass = all(ok for _, ok in checks)

    for label, ok in checks:
        icon = "✅" if ok else "❌"
        status = "[green]PASS[/]" if ok else "[red]FAIL[/]"
        console.print(f"  {icon}  {label}  →  {status}")

    console.print(f"\n  [dim]Combined trades/yr (BTC + ETH): {total_combined_per_yr:.0f}/yr[/]")

    console.print()
    if all_pass:
        console.print("[bold green]  ██████  PHASE 2 APPROVED — Proceed with live integration  ██████[/]")
        console.print("[green]  Add 30m scanner to main.py. Estimated combined trades: "
                      f"{total_combined_per_yr:.0f}/yr[/]")
    else:
        console.print("[bold red]  ██████  PHASE 2 BLOCKED — Do not integrate 30m yet  ██████[/]")
        console.print("[yellow]  Consider Option B (more assets) instead of Option C.[/]")
        failed = [label for label, ok in checks if not ok]
        console.print(f"[red]  Failed criteria: {failed}[/]")

    console.print()


if __name__ == "__main__":
    main()
