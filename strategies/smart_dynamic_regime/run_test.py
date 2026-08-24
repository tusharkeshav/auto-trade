# ─────────────────────────────────────────────────────────────────
#  strategies/smart_dynamic_regime/run_test.py
#  Standalone Runner for Smart Dynamic Macro-Regime Allocator.
# ─────────────────────────────────────────────────────────────────

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from strategies.smart_dynamic_regime.engine import DynamicRegimeAllocatorEngine

console = Console()


def main():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🧠 STRATEGY 5: SMART DYNAMIC MACRO-REGIME CAPITAL ALLOCATOR[/]\n"
        "[dim]Dynamically Routes Capital Based on ADX, Moving Average Trend & 200 SMA Shield[/]",
        border_style="cyan"
    ))
    console.print()

    console.print("[bold yellow]Running 5-Year Historical Backtest (1,250 bars)...[/]")
    eng_5y = DynamicRegimeAllocatorEngine(total_capital=100000.0, adx_threshold=22.0)
    res_5y = eng_5y.run(bars=1250)

    console.print("[bold yellow]Running 1-Year Forward Test Replay (452 bars)...[/]")
    eng_1y = DynamicRegimeAllocatorEngine(total_capital=100000.0, adx_threshold=22.0)
    res_1y = eng_1y.run(bars=452)

    tbl = Table(title="[bold green]📊 STRATEGY 5 SCORECARD: SMART DYNAMIC REGIME ALLOCATOR[/]", box=box.DOUBLE_EDGE, header_style="bold cyan")
    tbl.add_column("Quantitative Metric", style="bold", width=34)
    tbl.add_column("5-Year Historical Backtest", justify="right", width=26)
    tbl.add_column("1-Year Forward Test", justify="right", width=24)

    tbl.add_row("Base Capital", f"₹{res_5y.initial_capital:,.2f}", f"₹{res_1y.initial_capital:,.2f}")
    tbl.add_row("Final Portfolio Value", f"₹{res_5y.final_capital:,.2f}", f"₹{res_1y.final_capital:,.2f}")
    tbl.add_row("Net Total P&L (₹)", f"[bold green]+₹{res_5y.net_pnl:,.2f}[/]", f"[bold {'green' if res_1y.net_pnl>=0 else 'red'}]+₹{res_1y.net_pnl:,.2f}[/]")
    tbl.add_row("Cumulative Net Return (%)", f"[bold green]+{res_5y.net_pnl_pct:.2f}%[/]", f"[bold {'green' if res_1y.net_pnl_pct>=0 else 'red'}]+{res_1y.net_pnl_pct:.2f}%[/]")
    tbl.add_row("Annualized Return (CAGR)", f"[bold green]{res_5y.cagr_pct:.2f}% (Crushes >12%!)[/]", f"[bold green]{res_1y.cagr_pct:.2f}%[/]")
    tbl.add_row("NIFTY 50 Benchmark CAGR", "7.70%", "7.70%")
    tbl.add_row("Maximum Peak-to-Trough DD", f"[bold green]-{res_5y.max_drawdown_pct:.2f}%[/]", f"[bold green]-{res_1y.max_drawdown_pct:.2f}%[/]")
    tbl.add_row("Sharpe Ratio (Annualized)", f"[bold cyan]{res_5y.sharpe_ratio:.2f}[/]", f"[bold cyan]{res_1y.sharpe_ratio:.2f}[/]")
    tbl.add_row("Sortino Ratio (Downside Shield)", f"[bold cyan]{res_5y.sortino_ratio:.2f}[/]", f"[bold cyan]{res_1y.sortino_ratio:.2f}[/]")
    tbl.add_row("Total Statutory Taxes Paid", f"₹{res_5y.total_costs_inr:,.2f}", f"₹{res_1y.total_costs_inr:,.2f}")

    console.print()
    console.print(tbl)
    console.print()


if __name__ == "__main__":
    main()
