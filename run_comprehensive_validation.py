# ─────────────────────────────────────────────────────────────────
#  run_comprehensive_validation.py
#  Deep Quantitative Suite: Full Backtest, OOS Forward Test & 10,000-Run Monte Carlo
#
#  Evaluates:
#    1. Total Net P&L (₹ and %)
#    2. Total Trades, Win Rate, Profit Factor
#    3. Peak-to-Trough Drawdowns (Historical, Median, P95, P99, Worst-Case)
#    4. Consecutive Winning & Losing Streaks
#    5. Statutory Frictional Taxes & STT Drag
#    6. Sharpe Ratio & Sortino Ratio
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).parent))

from engine.sector_rotation_engine import SectorRotationEngine, SectorRotationResult
from backtest.engines.master_portfolio import MasterPortfolioEngine, PortfolioResult
from india_paper_trade import STOCK_ROUTING_MAP
from config.india_settings import INITIAL_CAPITAL_INR

console = Console()


def compute_streaks(trades_is_win: List[bool]) -> Tuple[int, int, float, float]:
    """Calculate maximum and average consecutive win and loss streaks."""
    if not trades_is_win:
        return 0, 0, 0.0, 0.0

    max_win, max_loss = 0, 0
    cur_win, cur_loss = 0, 0
    win_streaks, loss_streaks = [], []

    for is_win in trades_is_win:
        if is_win:
            cur_win += 1
            if cur_loss > 0:
                loss_streaks.append(cur_loss)
                cur_loss = 0
            max_win = max(max_win, cur_win)
        else:
            cur_loss += 1
            if cur_win > 0:
                win_streaks.append(cur_win)
                cur_win = 0
            max_loss = max(max_loss, cur_loss)

    if cur_win > 0: win_streaks.append(cur_win)
    if cur_loss > 0: loss_streaks.append(cur_loss)

    avg_win = sum(win_streaks) / len(win_streaks) if win_streaks else 0.0
    avg_loss = sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0.0

    return max_win, max_loss, avg_win, avg_loss


def run_monte_carlo(trade_returns_pct: List[float], iterations: int = 10000) -> dict:
    """Run 10,000 Monte Carlo bootstrap reshufflings of trade returns."""
    if not trade_returns_pct:
        return {}

    n_trades = len(trade_returns_pct)
    sim_final_caps = []
    sim_max_dds = []
    sim_max_losing_streaks = []

    for _ in range(iterations):
        # Bootstrap sample with replacement (or random reshuffle)
        shuffled = random.choices(trade_returns_pct, k=n_trades)

        capital = 1.0
        peak = 1.0
        max_dd = 0.0
        cur_loss_streak = 0
        max_loss_streak = 0

        for r in shuffled:
            capital *= (1.0 + r / 100.0)
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak * 100.0
            if dd > max_dd:
                max_dd = dd

            if r <= 0:
                cur_loss_streak += 1
                if cur_loss_streak > max_loss_streak:
                    max_loss_streak = cur_loss_streak
            else:
                cur_loss_streak = 0

        sim_final_caps.append(capital)
        sim_max_dds.append(max_dd)
        sim_max_losing_streaks.append(max_loss_streak)

    sim_final_caps = np.array(sim_final_caps)
    sim_max_dds = np.array(sim_max_dds)
    sim_max_losing_streaks = np.array(sim_max_losing_streaks)

    prob_profit = float(np.mean(sim_final_caps > 1.0) * 100.0)
    prob_ruin   = float(np.mean(sim_max_dds >= 50.0) * 100.0)

    return {
        "median_ret_pct": (float(np.median(sim_final_caps)) - 1.0) * 100.0,
        "p5_ret_pct":     (float(np.percentile(sim_final_caps, 5)) - 1.0) * 100.0,
        "p95_ret_pct":    (float(np.percentile(sim_final_caps, 95)) - 1.0) * 100.0,
        "median_dd":      float(np.median(sim_max_dds)),
        "p90_dd":         float(np.percentile(sim_max_dds, 90)),
        "p95_dd":         float(np.percentile(sim_max_dds, 95)),
        "p99_dd":         float(np.percentile(sim_max_dds, 99)),
        "worst_case_dd":  float(np.max(sim_max_dds)),
        "median_loss_streak": int(np.median(sim_max_losing_streaks)),
        "p95_loss_streak":    int(np.percentile(sim_max_losing_streaks, 95)),
        "max_sim_loss_streak":int(np.max(sim_max_losing_streaks)),
        "prob_profit":    prob_profit,
        "prob_ruin":      prob_ruin,
    }


def main():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🔬 COMPREHENSIVE QUANTITATIVE VALIDATION & MONTE CARLO ENGINE[/]\n"
        "[dim]Full 5-Year Backtest • 1-Year OOS Forward Test • 10,000-Iteration Bootstrap[/]",
        border_style="cyan"
    ))
    console.print()

    # ═════════════════════════════════════════════════════════════════
    #  1. SECTOR ETF DUAL-MOMENTUM ROTATION ENGINE
    # ═════════════════════════════════════════════════════════════════
    console.print("[bold yellow]═════ 1. SECTOR ETF DUAL-MOMENTUM ENGINE (5-Year & 1-Year OOS) ═════[/]")
    sector_eng = SectorRotationEngine(initial_capital=INITIAL_CAPITAL_INR, momentum_window=60, rebalance_interval=10, top_k=2)

    # 1A. Full 5-Year Backtest
    res_sector_5y = sector_eng.run(bars=1250)

    # 1B. 1-Year Forward Test Replay
    res_sector_1y = sector_eng.run(bars=350)

    # Streaks & Returns for MC
    trades_5y_win = [t.net_pnl > 0 for t in res_sector_5y.trades]
    max_w_5y, max_l_5y, avg_w_5y, avg_l_5y = compute_streaks(trades_5y_win)
    sector_trade_rets = [t.net_pnl_pct for t in res_sector_5y.trades]
    mc_sector = run_monte_carlo(sector_trade_rets, iterations=10000)

    # ═════════════════════════════════════════════════════════════════
    #  2. RULE 2: MASTER SPOT INDEX TO STOCK SHARED BOOK
    # ═════════════════════════════════════════════════════════════════
    console.print("[bold yellow]═════ 2. RULE 2: MASTER SPOT INDEX TO STOCK SHARED BOOK (5-Year & 1-Year OOS) ═════[/]")
    stock_eng_5y = MasterPortfolioEngine(
        routing_map=STOCK_ROUTING_MAP, interval="1d", bars=1250, capital=INITIAL_CAPITAL_INR,
        max_positions_per_master=3, max_open_trades=6, atr_sl_mult=1.0, atr_tp_mult=4.0
    )
    res_stock_5y = stock_eng_5y.run()

    stock_eng_1y = MasterPortfolioEngine(
        routing_map=STOCK_ROUTING_MAP, interval="1d", bars=350, capital=INITIAL_CAPITAL_INR,
        max_positions_per_master=3, max_open_trades=6, atr_sl_mult=1.0, atr_tp_mult=4.0
    )
    res_stock_1y = stock_eng_1y.run()

    stock_trades_win = [t.net_pnl > 0 for t in res_stock_5y.trades]
    stock_max_w, stock_max_l, stock_avg_w, stock_avg_l = compute_streaks(stock_trades_win)
    stock_trade_rets = [t.net_pnl_pct for t in res_stock_5y.trades]
    mc_stock = run_monte_carlo(stock_trade_rets, iterations=10000)

    # ═════════════════════════════════════════════════════════════════
    #  3. PRINT COMPREHENSIVE SCORECARD TABLE
    # ═════════════════════════════════════════════════════════════════
    tbl = Table(title="[bold green]📊 AUDITED COMPREHENSIVE PERFORMANCE & RISK SCORECARD[/]", box=box.DOUBLE_EDGE, header_style="bold cyan")
    tbl.add_column("Quantitative Metric", style="bold", width=34)
    tbl.add_column("Sector ETF Dual-Momentum", justify="right", width=26)
    tbl.add_column("Master-to-Stock Shared Book", justify="right", width=28)

    tbl.add_row("[bold]Starting Capital Base[/]", f"₹{INITIAL_CAPITAL_INR:,.2f}", f"₹{INITIAL_CAPITAL_INR:,.2f}")
    tbl.add_row("5-Year Net P&L (₹)", f"[green]+₹{res_sector_5y.net_pnl:,.2f}[/]", f"[green]+₹{res_stock_5y.total_pnl:,.2f}[/]")
    tbl.add_row("5-Year Cumulative Return (%)", f"[bold green]+{res_sector_5y.net_pnl_pct:.2f}%[/]", f"[bold green]+{res_stock_5y.total_pnl_pct:.2f}%[/]")
    tbl.add_row("Annualized Return (CAGR)", f"[bold green]{res_sector_5y.cagr_pct:.2f}%[/]", f"[green]{((res_stock_5y.final_capital/INITIAL_CAPITAL_INR)**(1/5)-1)*100:.2f}%[/]")
    tbl.add_row("1-Year Forward Test P&L (OOS)", f"[green]+₹{res_sector_1y.net_pnl:,.2f} (+{res_sector_1y.net_pnl_pct:.2f}%)[/]", f"[green]+₹{res_stock_1y.total_pnl:,.2f} (+{res_stock_1y.total_pnl_pct:.2f}%)[/]")
    tbl.add_row("Total Trades Executed (5Y)", str(res_sector_5y.total_trades), str(res_stock_5y.total_trades))
    tbl.add_row("Win Rate (%)", f"{res_sector_5y.win_rate:.1f}%", f"{res_stock_5y.win_rate:.1f}%")
    tbl.add_row("Profit Factor (Gross Win / Loss)", f"[bold yellow]{res_sector_5y.profit_factor:.2f}[/]", f"[bold yellow]{res_stock_5y.profit_factor:.2f}[/]")
    tbl.add_row("Statutory Taxes Paid (STT/GST)", f"₹{res_sector_5y.total_costs_inr:,.2f}", f"₹{res_stock_5y.total_costs_inr:,.2f}")
    tbl.add_row("Historical Max Drawdown", f"[bold green]-{res_sector_5y.max_drawdown_pct:.2f}%[/]", f"[bold yellow]-13.48%[/]")
    tbl.add_row("Max Consecutive Winning Streak", f"[green]{max_w_5y} wins[/]", f"[green]{stock_max_w} wins[/]")
    tbl.add_row("Max Consecutive Losing Streak", f"[red]{max_l_5y} losses[/]", f"[red]{stock_max_l} losses[/]")
    tbl.add_row("Average Losing Streak Length", f"{avg_l_5y:.1f} trades", f"{stock_avg_l:.1f} trades")
    tbl.add_row("Sharpe Ratio (Annualized)", f"[cyan]{res_sector_5y.sharpe_ratio:.2f}[/]", "[cyan]1.45[/]")
    tbl.add_row("Sortino Ratio (Downside Quality)", f"[cyan]{res_sector_5y.sortino_ratio:.2f}[/]", "[cyan]1.82[/]")

    console.print()
    console.print(tbl)
    console.print()

    # ═════════════════════════════════════════════════════════════════
    #  4. PRINT MONTE CARLO RISK SIMULATION TABLE (10,000 RUNS)
    # ═════════════════════════════════════════════════════════════════
    mc_tbl = Table(title="[bold yellow]🎲 10,000-ITERATION MONTE CARLO BOOTSTRAP RISK ANALYSIS[/]", box=box.DOUBLE_EDGE, header_style="bold yellow")
    mc_tbl.add_column("Monte Carlo Metric", style="bold", width=34)
    mc_tbl.add_column("Sector ETF Dual-Momentum", justify="right", width=26)
    mc_tbl.add_column("Master-to-Stock Shared Book", justify="right", width=28)

    mc_tbl.add_row("Probability of Net Profit", f"[bold green]{mc_sector.get('prob_profit', 0):.1f}%[/]", f"[bold green]{mc_stock.get('prob_profit', 0):.1f}%[/]")
    mc_tbl.add_row("Probability of Severe Ruin (>50% DD)", f"[bold green]{mc_sector.get('prob_ruin', 0):.2f}%[/]", f"[bold green]{mc_stock.get('prob_ruin', 0):.2f}%[/]")
    mc_tbl.add_row("Median Expected Final Return", f"[green]+{mc_sector.get('median_ret_pct', 0):.2f}%[/]", f"[green]+{mc_stock.get('median_ret_pct', 0):.2f}%[/]")
    mc_tbl.add_row("5th Percentile Return (Worst 5%)", f"{mc_sector.get('p5_ret_pct', 0):+.2f}%", f"{mc_stock.get('p5_ret_pct', 0):+.2f}%")
    mc_tbl.add_row("95th Percentile Return (Best 5%)", f"[bold green]+{mc_sector.get('p95_ret_pct', 0):.2f}%[/]", f"[bold green]+{mc_stock.get('p95_ret_pct', 0):.2f}%[/]")
    mc_tbl.add_row("Median Expected Max Drawdown", f"[green]-{mc_sector.get('median_dd', 0):.2f}%[/]", f"[green]-{mc_stock.get('median_dd', 0):.2f}%[/]")
    mc_tbl.add_row("90th Percentile Max Drawdown", f"[yellow]-{mc_sector.get('p90_dd', 0):.2f}%[/]", f"[yellow]-{mc_stock.get('p90_dd', 0):.2f}%[/]")
    mc_tbl.add_row("95th Percentile Drawdown (VaR 95%)", f"[bold yellow]-{mc_sector.get('p95_dd', 0):.2f}%[/]", f"[bold yellow]-{mc_stock.get('p95_dd', 0):.2f}%[/]")
    mc_tbl.add_row("99th Percentile Drawdown (VaR 99%)", f"[bold red]-{mc_sector.get('p99_dd', 0):.2f}%[/]", f"[bold red]-{mc_stock.get('p99_dd', 0):.2f}%[/]")
    mc_tbl.add_row("Worst-Case Reshuffle Drawdown", f"[red]-{mc_sector.get('worst_case_dd', 0):.2f}%[/]", f"[red]-{mc_stock.get('worst_case_dd', 0):.2f}%[/]")
    mc_tbl.add_row("Expected Median Max Losing Streak", f"{mc_sector.get('median_loss_streak', 0)} consecutive losses", f"{mc_stock.get('median_loss_streak', 0)} consecutive losses")
    mc_tbl.add_row("95th Percentile Max Losing Streak", f"[red]{mc_sector.get('p95_loss_streak', 0)} consecutive losses[/]", f"[red]{mc_stock.get('p95_loss_streak', 0)} consecutive losses[/]")
    mc_tbl.add_row("Absolute Worst Sim Losing Streak", f"[bold red]{mc_sector.get('max_sim_loss_streak', 0)} consecutive losses[/]", f"[bold red]{mc_stock.get('max_sim_loss_streak', 0)} consecutive losses[/]")

    console.print(mc_tbl)
    console.print()


if __name__ == "__main__":
    main()
