# ─────────────────────────────────────────────────────────────────
#  run_sector_rotation_backtest.py
#  Standalone CLI Runner for Sector ETF Dual-Momentum Rotation Engine.
#
#  Usage:
#      source .venv/bin/activate && python run_sector_rotation_backtest.py
#      python run_sector_rotation_backtest.py --bars 1250 --top-k 2 --rebalance-days 10
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

sys.path.insert(0, str(Path(__file__).parent))

from engine.sector_rotation_engine import SectorRotationEngine
from config.india_settings import INITIAL_CAPITAL_INR

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Sector ETF Dual-Momentum Backtest Runner")
    parser.add_argument("--bars", type=int, default=1250, help="Number of daily trading bars (~5 years = 1250)")
    parser.add_argument("--top-k", type=int, default=2, help="Number of top sectors to allocate capital to")
    parser.add_argument("--rebalance-days", type=int, default=10, help="Rebalancing frequency in trading days (default: 10 = bi-weekly)")
    parser.add_argument("--momentum-window", type=int, default=60, help="Lookback window for momentum return calculation (default: 60)")
    parser.add_argument("--no-shield", action="store_true", help="Disable 200 SMA Macro Cash Shield")
    args = parser.parse_args()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]🚀 INSTITUTIONAL SECTOR ETF DUAL-MOMENTUM ROTATION ENGINE[/]\n"
        "[dim]Gary Antonacci Model • 200 SMA Macro Cash Shield • Real Indian CNC Statutory Tax[/]",
        border_style="cyan"
    ))
    console.print()

    engine = SectorRotationEngine(
        initial_capital    = INITIAL_CAPITAL_INR,
        momentum_window    = args.momentum_window,
        rebalance_interval = args.rebalance_days,
        top_k              = args.top_k,
        use_macro_shield   = not args.no_shield,
    )

    result = engine.run(bars=args.bars)

    # ── 1. Top KPI Summary Table ─────────────────────────────────────
    kpi_tbl = Table(title="[bold green]📊 AUDITED STRATEGY VS NIFTY 50 BENCHMARK[/]", box=box.DOUBLE_EDGE, header_style="bold cyan")
    kpi_tbl.add_column("Performance Metric", style="bold", width=32)
    kpi_tbl.add_column("Sector Dual-Momentum", justify="right", width=24)
    kpi_tbl.add_column("NIFTY 50 Benchmark", justify="right", width=22)

    ret_col = "green" if result.net_pnl >= 0 else "red"
    ret_sgn = "+" if result.net_pnl >= 0 else ""

    kpi_tbl.add_row("Starting Benchmark Capital", f"₹{result.initial_capital:,.2f}", f"₹{result.initial_capital:,.2f}")
    kpi_tbl.add_row("Final Audited Portfolio Value", f"₹{result.final_capital:,.2f}", "—")
    kpi_tbl.add_row("Net Total P&L (₹)", f"[{ret_col}]{ret_sgn}₹{result.net_pnl:,.2f}[/]", "—")
    kpi_tbl.add_row("Cumulative Net Return (%)", f"[{ret_col}]{ret_sgn}{result.net_pnl_pct:.2f}%[/]", "—")
    kpi_tbl.add_row("Annualized Return (CAGR)", f"[bold green]{result.cagr_pct:.2f}%[/]", f"[yellow]{result.benchmark_cagr:.2f}%[/]")
    kpi_tbl.add_row("Maximum Peak-to-Trough Drawdown", f"[bold green]-{result.max_drawdown_pct:.2f}%[/]", f"[bold red]-{result.benchmark_dd:.2f}%[/]")
    kpi_tbl.add_row("Sharpe Ratio (Annualized)", f"[bold cyan]{result.sharpe_ratio:.2f}[/]", "—")
    kpi_tbl.add_row("Sortino Ratio (Downside Vol)", f"[bold cyan]{result.sortino_ratio:.2f}[/]", "—")
    kpi_tbl.add_row("Profit Factor (Gross Win / Loss)", f"[bold yellow]{result.profit_factor:.2f}[/]", "—")
    kpi_tbl.add_row("Trade Rotation Win Rate", f"{result.win_rate:.1f}% ({result.winning_trades}W / {result.losing_trades}L)", "—")
    kpi_tbl.add_row("Total Rotations Executed", str(result.total_trades), "Buy & Hold (1)")
    kpi_tbl.add_row("Total Statutory Taxes Paid (STT/GST)", f"[dim]₹{result.total_costs_inr:,.2f}[/]", "—")
    kpi_tbl.add_row("Tax Frictional Cost Drag", f"[dim]{(result.total_costs_inr / (result.net_pnl + result.total_costs_inr) * 100.0) if (result.net_pnl + result.total_costs_inr) > 0 else 0.0:.2f}% of Gross Profit[/]", "—")

    console.print(kpi_tbl)
    console.print()

    # ── 2. Recent Rebalance Log ──────────────────────────────────────
    if result.rebalance_log:
        reb_tbl = Table(title="[bold yellow]🔄 RECENT REBALANCE HISTORY (LAST 8 PERIODS)[/]", box=box.SIMPLE_HEAD, header_style="bold yellow")
        reb_tbl.add_column("Rebalance Date", width=14)
        reb_tbl.add_column("Regime & Strategy Status", width=42)
        reb_tbl.add_column("Allocated ETF Holdings", width=32)
        reb_tbl.add_column("Portfolio NAV (₹)", justify="right", width=18)
        reb_tbl.add_column("Liquid Cash (₹)", justify="right", width=16)

        for log_entry in result.rebalance_log[-8:]:
            reb_tbl.add_row(
                log_entry["date"],
                log_entry["regime"],
                ", ".join(log_entry["holdings"]) if log_entry["holdings"] else "Cash / Safe Asset",
                f"₹{log_entry['portfolio_value']:,.2f}",
                f"₹{log_entry['cash']:,.2f}",
            )
        console.print(reb_tbl)
        console.print()

    # ── 3. Trade Rotations Log ───────────────────────────────────────
    if result.trades:
        tr_tbl = Table(title="[bold cyan]📜 RECENT COMPLETED TRADE ROTATIONS (LAST 10 TRADES)[/]", box=box.SIMPLE_HEAD, header_style="bold cyan")
        tr_tbl.add_column("Entry Date", width=12)
        tr_tbl.add_column("Exit Date", width=12)
        tr_tbl.add_column("ETF Symbol", width=14)
        tr_tbl.add_column("Entry (₹)", justify="right", width=10)
        tr_tbl.add_column("Exit (₹)", justify="right", width=10)
        tr_tbl.add_column("Bars Held", justify="right", width=10)
        tr_tbl.add_column("Taxes (₹)", justify="right", width=10)
        tr_tbl.add_column("Net P&L (₹)", justify="right", width=14)
        tr_tbl.add_column("Return (%)", justify="right", width=12)

        for t in result.trades[-10:]:
            pnl_col = "green" if t.net_pnl >= 0 else "red"
            pnl_sgn = "+" if t.net_pnl >= 0 else ""
            tr_tbl.add_row(
                t.entry_date.strftime("%Y-%m-%d"),
                t.exit_date.strftime("%Y-%m-%d"),
                f"[bold]{t.symbol}[/]",
                f"₹{t.entry_price:,.2f}",
                f"₹{t.exit_price:,.2f}",
                f"{t.bars_held}d",
                f"₹{t.cost_inr:,.2f}",
                f"[{pnl_col}]{pnl_sgn}₹{t.net_pnl:,.2f}[/]",
                f"[{pnl_col}]{pnl_sgn}{t.net_pnl_pct:.2f}%[/]",
            )
        console.print(tr_tbl)
        console.print()


if __name__ == "__main__":
    main()
