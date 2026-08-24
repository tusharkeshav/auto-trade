# ─────────────────────────────────────────────────────────────────
#  strategies/sector_rotation/run_test.py
#  Standalone Runner for Sector ETF Dual-Momentum Engine.
# ─────────────────────────────────────────────────────────────────

import random
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from strategies.sector_rotation.engine import SectorRotationEngine

console = Console()


def run_monte_carlo(trade_returns_pct: list[float], iterations: int = 10000) -> dict:
    if not trade_returns_pct: return {}
    n = len(trade_returns_pct)
    sim_caps, sim_dds = [], []
    for _ in range(iterations):
        shuffled = random.choices(trade_returns_pct, k=n)
        cap = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in shuffled:
            cap *= (1.0 + r / 100.0)
            if cap > peak: peak = cap
            dd = (peak - cap) / peak * 100.0
            if dd > max_dd: max_dd = dd
        sim_caps.append(cap)
        sim_dds.append(max_dd)

    sim_caps, sim_dds = np.array(sim_caps), np.array(sim_dds)
    return {
        "prob_profit": float(np.mean(sim_caps > 1.0) * 100.0),
        "median_ret":  (float(np.median(sim_caps)) - 1.0) * 100.0,
        "median_dd":   float(np.median(sim_dds)),
        "p95_dd":      float(np.percentile(sim_dds, 95)),
    }


def main():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]📈 STRATEGY 1: SECTOR ETF DUAL-MOMENTUM ROTATION ENGINE[/]\n"
        "[dim]Auditing 5-Year Backtest (2021–2026), 1-Year Forward Test & 10,000-Run Monte Carlo[/]",
        border_style="cyan"
    ))
    console.print()

    console.print("[bold yellow]Running 5-Year Historical Backtest (1,250 bars)...[/]")
    eng_5y = SectorRotationEngine(initial_capital=50000.0, top_k=2, rebalance_interval=10)
    res_5y = eng_5y.run(bars=1250)

    console.print("[bold yellow]Running 1-Year Forward Test Replay (452 bars)...[/]")
    eng_1y = SectorRotationEngine(initial_capital=50000.0, top_k=2, rebalance_interval=10)
    res_1y = eng_1y.run(bars=452)

    mc_stats = run_monte_carlo([t.net_pnl_pct for t in res_5y.trades])

    tbl = Table(title="[bold green]📊 STRATEGY 1 SCORECARD: SECTOR ETF ROTATION[/]", box=box.DOUBLE_EDGE, header_style="bold cyan")
    tbl.add_column("Quantitative Metric", style="bold", width=34)
    tbl.add_column("5-Year Historical Backtest", justify="right", width=26)
    tbl.add_column("1-Year Forward Test", justify="right", width=24)

    tbl.add_row("Base Capital", f"₹{res_5y.initial_capital:,.2f}", f"₹{res_1y.initial_capital:,.2f}")
    tbl.add_row("Final Portfolio Value", f"₹{res_5y.final_capital:,.2f}", f"₹{res_1y.final_capital:,.2f}")
    tbl.add_row("Net Total P&L (₹)", f"[bold green]+₹{res_5y.net_pnl:,.2f}[/]", f"[bold {'green' if res_1y.net_pnl>=0 else 'red'}]{'+' if res_1y.net_pnl>=0 else ''}₹{res_1y.net_pnl:,.2f}[/]")
    tbl.add_row("Cumulative Net Return (%)", f"[bold green]+{res_5y.net_pnl_pct:.2f}%[/]", f"[bold {'green' if res_1y.net_pnl_pct>=0 else 'red'}]{'+' if res_1y.net_pnl_pct>=0 else ''}{res_1y.net_pnl_pct:.2f}%[/]")
    tbl.add_row("Annualized Return (CAGR)", f"[bold green]{res_5y.cagr_pct:.2f}%[/]", f"{res_1y.cagr_pct:.2f}%")
    tbl.add_row("Maximum Peak-to-Trough DD", f"[bold green]-{res_5y.max_drawdown_pct:.2f}%[/]", f"-{res_1y.max_drawdown_pct:.2f}%")
    tbl.add_row("Profit Factor (Gross Win / Loss)", f"[bold yellow]{res_5y.profit_factor:.2f}[/]", f"{res_1y.profit_factor:.2f}")
    tbl.add_row("Audited Win Rate (%)", f"{res_5y.win_rate:.1f}%", f"{res_1y.win_rate:.1f}%")
    tbl.add_row("Total Trades Executed", f"{res_5y.total_trades} trades (~10/yr)", f"{res_1y.total_trades} trades")
    tbl.add_row("Sharpe Ratio (Annualized)", f"[bold cyan]{res_5y.sharpe_ratio:.2f}[/]", f"{res_1y.sharpe_ratio:.2f}")
    tbl.add_row("Sortino Ratio (Downside Shield)", f"[bold cyan]{res_5y.sortino_ratio:.2f}[/]", f"{res_1y.sortino_ratio:.2f}")
    tbl.add_row("Total Statutory Taxes Paid", f"₹{res_5y.total_costs_inr:,.2f}", f"₹{res_1y.total_costs_inr:,.2f}")
    tbl.add_row("10k Monte Carlo Prob of Profit", f"[bold green]{mc_stats.get('prob_profit',0):.1f}%[/]", "—")
    tbl.add_row("Monte Carlo 95% VaR Drawdown", f"-{mc_stats.get('p95_dd',0):.2f}%", "—")

    console.print()
    console.print(tbl)
    console.print()


if __name__ == "__main__":
    main()
