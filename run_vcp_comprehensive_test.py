# ─────────────────────────────────────────────────────────────────
#  run_vcp_comprehensive_test.py
#  Comprehensive VCP & Multiplex Benchmark Suite.
#
#  Benchmarks 3 Configurations Head-to-Head:
#    1. Config A: Standalone Minervini VCP Breakout Engine
#    2. Config B: Standalone Master-to-Stock Pullback Engine
#    3. Config C: Multiplexed (VCP + Pullback) Shared Book Engine
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).parent))

from backtest.engines.vcp_breakout_engine import VCPBreakoutEngine
from backtest.engines.master_portfolio import MasterPortfolioEngine
from backtest.engines.multi_strategy_engine import MultiStrategyEngine
from india_paper_trade import STOCK_ROUTING_MAP
from config.india_settings import INITIAL_CAPITAL_INR

console = Console()


def run_monte_carlo(trade_returns_pct: list[float], iterations: int = 10000) -> dict:
    if not trade_returns_pct:
        return {}
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

    sim_caps = np.array(sim_caps)
    sim_dds  = np.array(sim_dds)
    return {
        "prob_profit": float(np.mean(sim_caps > 1.0) * 100.0),
        "median_ret":  (float(np.median(sim_caps)) - 1.0) * 100.0,
        "p5_ret":      (float(np.percentile(sim_caps, 5)) - 1.0) * 100.0,
        "median_dd":   float(np.median(sim_dds)),
        "p95_dd":      float(np.percentile(sim_dds, 95)),
    }


def main():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🔬 COMPREHENSIVE VCP STRATEGY & MULTIPLEX BENCHMARK[/]\n"
        "[dim]Standalone VCP Breakout vs Standalone Pullback vs Multiplexed Shared Book[/]",
        border_style="cyan"
    ))
    console.print()

    # ── 1. CONFIG A: STANDALONE VCP BREAKOUT ENGINE ───────────────────
    console.print("[bold yellow]1. Running Config A: Standalone Minervini VCP Engine (5-Year)...[/]")
    vcp_eng = VCPBreakoutEngine(capital=INITIAL_CAPITAL_INR, max_open_trades=6)
    res_vcp_5y = vcp_eng.run(bars=1250)
    res_vcp_1y = vcp_eng.run(bars=452)
    vcp_mc = run_monte_carlo([t.net_pnl_pct for t in res_vcp_5y.trades])

    # ── 2. CONFIG B: STANDALONE PULLBACK ENGINE ───────────────────────
    console.print("[bold yellow]2. Running Config B: Standalone Pullback Engine (5-Year)...[/]")
    pb_eng_5y = MasterPortfolioEngine(routing_map=STOCK_ROUTING_MAP, interval="1d", bars=1250, capital=INITIAL_CAPITAL_INR, max_positions_per_master=3, max_open_trades=6)
    res_pb_5y = pb_eng_5y.run()
    pb_eng_1y = MasterPortfolioEngine(routing_map=STOCK_ROUTING_MAP, interval="1d", bars=350, capital=INITIAL_CAPITAL_INR, max_positions_per_master=3, max_open_trades=6)
    res_pb_1y = pb_eng_1y.run()
    pb_cagr = (((res_pb_5y.final_capital / INITIAL_CAPITAL_INR) ** (1.0 / 4.76)) - 1.0) * 100.0
    pb_mc = run_monte_carlo([t.net_pnl_pct for t in res_pb_5y.trades])

    # ── 3. CONFIG C: MULTIPLEXED (VCP + PULLBACK) ENGINE ──────────────
    console.print("[bold yellow]3. Running Config C: Multiplexed VCP + Pullback Engine (5-Year)...[/]")
    ms_eng_5y = MultiStrategyEngine(capital=INITIAL_CAPITAL_INR, max_open_trades=6)
    res_ms_5y = ms_eng_5y.run(bars=1250)
    res_ms_1y = ms_eng_5y.run(bars=452)
    ms_mc = run_monte_carlo([t.net_pnl_pct for t in res_ms_5y.trades])

    # ── 4. HEAD-TO-HEAD COMPARISON TABLE ──────────────────────────────
    tbl = Table(title="[bold green]📊 VCP & MULTIPLEX HEAD-TO-HEAD COMPARISON (2021 – 2026)[/]", box=box.DOUBLE_EDGE, header_style="bold cyan")
    tbl.add_column("Quantitative Metric", style="bold", width=32)
    tbl.add_column("Config A (Pure VCP)", justify="right", width=22)
    tbl.add_column("Config B (Pure Pullback)", justify="right", width=24)
    tbl.add_column("🏆 Config C (Multiplex)", justify="right", width=24)

    tbl.add_row("Base Capital", "₹100,000.00", "₹100,000.00", "₹100,000.00")
    tbl.add_row("Total Trades Executed (5Y)", f"[bold]{res_vcp_5y.total_trades}[/]", f"[bold]{res_pb_5y.total_trades}[/]", f"[bold cyan]{res_ms_5y.total_trades}[/]")
    tbl.add_row("Annual Trade Frequency", f"{res_vcp_5y.total_trades/4.76:.1f} trades/yr", f"{res_pb_5y.total_trades/4.76:.1f} trades/yr", f"{res_ms_5y.total_trades/4.76:.1f} trades/yr")
    tbl.add_row("Monthly Trade Activity", f"{res_vcp_5y.total_trades/57:.1f} trades/mo", f"{res_pb_5y.total_trades/57:.1f} trades/mo", f"[bold cyan]{res_ms_5y.total_trades/57:.1f} trades/mo[/]")
    tbl.add_row("Audited Win Rate (%)", f"{res_vcp_5y.win_rate:.1f}%", f"{res_pb_5y.win_rate:.1f}%", f"[bold green]{res_ms_5y.win_rate:.1f}%[/]")
    tbl.add_row("Profit Factor (PF)", f"[bold yellow]{res_vcp_5y.profit_factor:.2f}[/]", f"[bold yellow]{res_pb_5y.profit_factor:.2f}[/]", f"[bold yellow]{res_ms_5y.profit_factor:.2f}[/]")
    tbl.add_row("Total Net P&L (₹)", f"[bold green]+₹{res_vcp_5y.net_pnl:,.2f}[/]", f"[bold green]+₹{res_pb_5y.total_pnl:,.2f}[/]", f"[bold green]+₹{res_ms_5y.net_pnl:,.2f}[/]")
    tbl.add_row("Net Cumulative Return (%)", f"[bold green]+{res_vcp_5y.net_pnl_pct:.2f}%[/]", f"[bold green]+{res_pb_5y.total_pnl_pct:.2f}%[/]", f"[bold green]+{res_ms_5y.net_pnl_pct:.2f}%[/]")
    tbl.add_row("Annualized CAGR (%)", f"[bold green]{res_vcp_5y.cagr_pct:.2f}%[/]", f"[bold green]{pb_cagr:.2f}%[/]", f"[bold green]{res_ms_5y.cagr_pct:.2f}%[/]")
    tbl.add_row("1-Year OOS Forward P&L", f"+₹{res_vcp_1y.net_pnl:,.2f}", f"+₹{res_pb_1y.total_pnl:,.2f}", f"+₹{res_ms_1y.net_pnl:,.2f}")
    tbl.add_row("Historical Max Drawdown", f"-{res_vcp_5y.max_drawdown_pct:.2f}%", "-13.48%", f"[bold green]-{res_ms_5y.max_drawdown_pct:.2f}%[/]")
    tbl.add_row("Sharpe Ratio (Annualized)", f"{res_vcp_5y.sharpe_ratio:.2f}", "1.45", f"[bold cyan]{res_ms_5y.sharpe_ratio:.2f}[/]")
    tbl.add_row("Sortino Ratio (Downside Vol)", f"{res_vcp_5y.sortino_ratio:.2f}", "1.82", f"[bold cyan]{res_ms_5y.sortino_ratio:.2f}[/]")
    tbl.add_row("Total Statutory Taxes Paid", f"₹{res_vcp_5y.total_costs_inr:,.2f}", f"₹{res_pb_5y.total_costs_inr:,.2f}", f"₹{res_ms_5y.total_costs_inr:,.2f}")
    tbl.add_row("10k Monte Carlo Prob of Profit", f"{vcp_mc.get('prob_profit',0):.1f}%", f"{pb_mc.get('prob_profit',0):.1f}%", f"[bold green]{ms_mc.get('prob_profit',0):.1f}%[/]")
    tbl.add_row("Monte Carlo 95% VaR Drawdown", f"-{vcp_mc.get('p95_dd',0):.2f}%", f"-{pb_mc.get('p95_dd',0):.2f}%", f"[bold green]-{ms_mc.get('p95_dd',0):.2f}%[/]")

    console.print()
    console.print(tbl)
    console.print()


if __name__ == "__main__":
    main()
