# ─────────────────────────────────────────────────────────────────
#  run_forward_replay.py  —  Out-of-Sample Forward Test Replay
#
#  Objective:
#    Simulate exact blind live paper trading across the last 60 days
#    (May 1, 2026 → July 4, 2026) to prove out-of-sample forward edge.
#
#  Strategy: Rule 2 (Master Spot Index → Stock CNC Cash Delivery)
#  Universe: Top 12 NIFTY Sector Leaders (Banking, IT, Industrial, Telecom, Energy, Auto, Pharma, FMCG, Finance)
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.engines.master_to_stock import MasterIndexToStockEngine, MasterStockResult, MasterStockTrade
from backtest.engines.master_portfolio import MasterPortfolioEngine
from config.india_settings import INITIAL_CAPITAL_INR, INDIA_SIGNAL_THRESHOLD

console = Console()
IST = ZoneInfo("Asia/Kolkata")

STOCK_ROUTING_MAP = {
    # ── Secular Mega-Cap Leaders -> NIFTY50 Macro Pullback Shield ──
    "INFY.NS":       "^NSEI",                 # IT
    "TCS.NS":        "^NSEI",                 # IT
    "LT.NS":         "^NSEI",                 # Infrastructure
    "BHARTIARTL.NS": "^NSEI",                 # Telecom
    "RELIANCE.NS":   "^NSEI",                 # Energy
    "NTPC.NS":       "^NSEI",                 # Energy
    "SUNPHARMA.NS":  "^NSEI",                 # Pharma
    "ITC.NS":        "^NSEI",                 # FMCG
}


def _cluster_report(trades: list[MasterStockTrade]) -> None:
    """Print portfolio concentration warning for same-day clustered entries."""
    from collections import Counter
    day_entries: dict[str, list[str]] = {}
    for t in trades:
        day_key = t.entry_time.strftime("%Y-%m-%d")
        day_entries.setdefault(day_key, []).append(t.symbol)

    clusters = {d: syms for d, syms in day_entries.items() if len(syms) >= 3}
    if not clusters:
        return

    console.print(f"\n[bold yellow]⚠ CONCENTRATION RISK: {len(clusters)} day(s) with 3+ simultaneous entries[/]")
    for day, syms in sorted(clusters.items()):
        masters = [STOCK_ROUTING_MAP.get(s, "?") for s in syms]
        master_counts = Counter(masters)
        master_str = ", ".join(f"{m}×{c}" for m, c in master_counts.items())
        console.print(f"  {day}: {len(syms)} entries ({', '.join(syms)}) — master exposure: {master_str}")


def run_forward_test(interval: str = "1d", years: float = 1.0):
    start_date = datetime.now(IST) - timedelta(days=int(365.25 * years))
    console.print(f"\n[bold cyan]── {years}-YEAR OUT-OF-SAMPLE FORWARD TEST REPLAY ({start_date.strftime('%b %Y')} – {datetime.now(IST).strftime('%b %Y')}, {interval}) ──[/]")
    console.print("[dim]Simulating blind live daemon execution | Pure CNC Delivery | 0% Leverage[/]\n")

    bars_to_fetch = int(250 * years) + 250 if interval == "1d" else int(1750 * years) + 1400
    slice_cap = INITIAL_CAPITAL_INR / len(STOCK_ROUTING_MAP)

    results: dict[str, MasterStockResult] = {}
    all_trades: list[MasterStockTrade] = []

    for stk, master in STOCK_ROUTING_MAP.items():
        logger.info(f"Replaying blind forward execution for {stk} via {master} ({interval})...")
        engine = MasterIndexToStockEngine(
            stock_symbol = stk,
            master_index = master,
            interval     = interval,
            bars         = bars_to_fetch,
            threshold    = INDIA_SIGNAL_THRESHOLD,
            atr_sl_mult  = 1.0,
            atr_tp_mult  = 3.0,
            capital      = slice_cap,
            vix          = 16.0,
        )
        res = engine.run()

        # Filter trades strictly to those entered on or after start_date
        forward_trades = [t for t in res.trades if t.entry_time >= start_date]
        res.trades = forward_trades
        res.final_capital = slice_cap + sum(t.net_pnl for t in forward_trades)

        results[stk] = res
        all_trades.extend(forward_trades)

    # Report concentration risk (don't fake-remove trades — engines ran independently)
    _cluster_report(all_trades)

    # ── Print Forward Test Report ────────────────────────────────────────
    tot_pnl  = sum(r.total_pnl for r in results.values())
    tot_cost = sum(r.total_costs_inr for r in results.values())
    tot_win  = len([t for t in all_trades if t.net_pnl > 0])
    tot_tr   = len(all_trades)
    win_rate = tot_win / tot_tr * 100 if tot_tr else 0.0
    gp = sum(t.net_pnl for t in all_trades if t.net_pnl > 0)
    gl = abs(sum(t.net_pnl for t in all_trades if t.net_pnl <= 0))
    pf = round(gp / gl, 2) if gl else float("inf")

    tbl = Table(box=box.DOUBLE_EDGE, show_header=True, header_style="bold cyan")
    tbl.add_column("Stock Symbol",  width=14)
    tbl.add_column("Master Index",  width=12)
    tbl.add_column("Forward Trades",justify="right", width=14)
    tbl.add_column("Win Rate",      justify="right", width=10)
    tbl.add_column("Profit Factor", justify="right", width=13)
    tbl.add_column("Taxes Paid (₹)",justify="right", width=14)
    tbl.add_column("Net P&L (₹)",   justify="right", width=14)
    tbl.add_column("Net Return (%)",justify="right", width=14)

    for sym, r in results.items():
        s_col = "green" if r.total_pnl >= 0 else "red"
        s_sgn = "+" if r.total_pnl >= 0 else ""
        s_pfc = "green" if r.profit_factor >= 1.5 else ("yellow" if r.profit_factor >= 1.0 else "red")
        tbl.add_row(
            f"[bold]{sym}[/]",
            f"[cyan]{r.master_index}[/]",
            str(r.total_trades),
            f"{r.win_rate:.1f}%",
            f"[{s_pfc}]{r.profit_factor:.2f}[/]",
            f"₹{r.total_costs_inr:>9,.2f}",
            f"[{s_col}]{s_sgn}₹{r.total_pnl:>9,.2f}[/]",
            f"[{s_col}]{s_sgn}{r.total_pnl_pct:>6.2f}%[/]",
        )
    console.print(tbl)

    console.print(f"\n[bold]{years}-YEAR FORWARD REPLAY SUMMARY ({interval})[/]: "
                  f"Trades={tot_tr} | Win Rate=[yellow]{win_rate:.1f}%[/] | "
                  f"PF=[cyan]{pf}[/] | Taxes=₹{tot_cost:,.2f} | "
                  f"Net P&L=[{'green' if tot_pnl>=0 else 'red'}]{'+' if tot_pnl>=0 else ''}₹{tot_pnl:,.2f} "
                  f"({'+' if tot_pnl>=0 else ''}{tot_pnl/INITIAL_CAPITAL_INR*100:.2f}%)[/]\n")

    if all_trades:
        console.print(f"[bold cyan]── DETAILED FORWARD TRADE LOG ({start_date.strftime('%b %Y')} – {datetime.now(IST).strftime('%b %Y')}, {interval}) ──[/]")
        log_tbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold yellow")
        log_tbl.add_column("Entry Date", width=12)
        log_tbl.add_column("Exit Date",  width=12)
        log_tbl.add_column("Stock",      width=14)
        log_tbl.add_column("Entry ₹",    justify="right", width=10)
        log_tbl.add_column("Exit ₹",     justify="right", width=10)
        log_tbl.add_column("Exit Type",  width=14)
        log_tbl.add_column("Net P&L (₹)",justify="right", width=12)
        log_tbl.add_column("Return (%)", justify="right", width=10)

        for t in sorted(all_trades, key=lambda x: x.entry_time):
            c_col = "green" if t.net_pnl >= 0 else "red"
            c_sgn = "+" if t.net_pnl >= 0 else ""
            log_tbl.add_row(
                t.entry_time.strftime("%Y-%m-%d"),
                t.exit_time.strftime("%Y-%m-%d"),
                f"[bold]{t.symbol}[/]",
                f"₹{t.entry_price:,.2f}",
                f"₹{t.exit_price:,.2f}",
                t.exit_type,
                f"[{c_col}]{c_sgn}₹{t.net_pnl:,.2f}[/]",
                f"[{c_col}]{c_sgn}{t.net_pnl_pct:.2f}%[/]",
            )
        console.print(log_tbl)
        console.print()


def run_portfolio_test(interval: str = "1d", max_positions_per_master: int = 3, max_open_trades: int = 6, years: float = 1.0):
    start_date = datetime.now(IST) - timedelta(days=int(365.25 * years))
    console.print(f"\n[bold magenta]── {years}-YEAR SHARED-BOOK PORTFOLIO REPLAY ({start_date.strftime('%b %Y')} – {datetime.now(IST).strftime('%b %Y')}, {interval}) ──[/]")
    console.print(f"[dim]Simulating shared ₹500k capital pool | Signal Conviction Priority | MAX_POSITIONS_PER_MASTER={max_positions_per_master} | MAX_OPEN_TRADES={max_open_trades}[/]\n")

    bars_to_fetch = int(250 * years) + 250 if interval == "1d" else int(1750 * years) + 1400
    engine = MasterPortfolioEngine(
        routing_map = STOCK_ROUTING_MAP,
        interval    = interval,
        bars        = bars_to_fetch,
        threshold   = INDIA_SIGNAL_THRESHOLD,
        atr_sl_mult = 1.0,
        atr_tp_mult = 4.0,
        capital     = INITIAL_CAPITAL_INR,
        vix         = 16.0,
        max_positions_per_master = max_positions_per_master,
        max_open_trades = max_open_trades
    )
    res = engine.run()

    # Filter trades strictly to those entered on or after start_date
    forward_trades = [t for t in res.trades if t.entry_time >= start_date]
    res.trades = forward_trades
    res.final_capital = INITIAL_CAPITAL_INR + sum(t.net_pnl for t in forward_trades)

    tot_pnl  = res.total_pnl
    tot_cost = res.total_costs_inr
    tot_win  = len(res.winning_trades)
    tot_tr   = res.total_trades
    win_rate = tot_win / tot_tr * 100 if tot_tr else 0.0
    pf       = res.profit_factor

    tbl = Table(box=box.DOUBLE_EDGE, show_header=True, header_style="bold magenta")
    tbl.add_column("Stock Symbol",  width=14)
    tbl.add_column("Master Index",  width=12)
    tbl.add_column("Forward Trades",justify="right", width=14)
    tbl.add_column("Win Rate",      justify="right", width=10)
    tbl.add_column("Profit Factor", justify="right", width=13)
    tbl.add_column("Taxes Paid (₹)",justify="right", width=14)
    tbl.add_column("Net P&L (₹)",   justify="right", width=14)
    tbl.add_column("Net Return (%)",justify="right", width=14)

    for sym, s_res in res.stock_results.items():
        s_trades = [t for t in s_res.trades if t.entry_time >= start_date]
        s_pnl = sum(t.net_pnl for t in s_trades)
        s_cost = sum(t.cost_inr for t in s_trades)
        s_win = len([t for t in s_trades if t.net_pnl > 0])
        s_tr = len(s_trades)
        s_wr = s_win / s_tr * 100 if s_tr else 0.0
        s_gp = sum(t.net_pnl for t in s_trades if t.net_pnl > 0)
        s_gl = abs(sum(t.net_pnl for t in s_trades if t.net_pnl <= 0))
        s_pf = round(s_gp / s_gl, 2) if s_gl else float("inf")
        s_ret = s_pnl / (INITIAL_CAPITAL_INR / len(STOCK_ROUTING_MAP)) * 100

        s_col = "green" if s_pnl >= 0 else "red"
        s_sgn = "+" if s_pnl >= 0 else ""
        s_pfc = "green" if s_pf >= 1.5 else ("yellow" if s_pf >= 1.0 else "red")
        tbl.add_row(
            f"[bold]{sym}[/]",
            f"[cyan]{s_res.master_index}[/]",
            str(s_tr),
            f"{s_wr:.1f}%",
            f"[{s_pfc}]{s_pf:.2f}[/]",
            f"₹{s_cost:>9,.2f}",
            f"[{s_col}]{s_sgn}₹{s_pnl:>9,.2f}[/]",
            f"[{s_col}]{s_sgn}{s_ret:>6.2f}%[/]",
        )
    console.print(tbl)

    console.print(f"\n[bold]{years}-YEAR SHARED PORTFOLIO SUMMARY ({interval})[/]: "
                  f"Trades={tot_tr} | Win Rate=[yellow]{win_rate:.1f}%[/] | "
                  f"PF=[cyan]{pf}[/] | Taxes=₹{tot_cost:,.2f} | "
                  f"Net P&L=[{'green' if tot_pnl>=0 else 'red'}]{'+' if tot_pnl>=0 else ''}₹{tot_pnl:,.2f} "
                  f"({'+' if tot_pnl>=0 else ''}{tot_pnl/INITIAL_CAPITAL_INR*100:.2f}%)[/]\n")

    if forward_trades:
        console.print(f"[bold magenta]── DETAILED PORTFOLIO TRADE LOG ({start_date.strftime('%b %Y')} – {datetime.now(IST).strftime('%b %Y')}, {interval}) ──[/]")
        log_tbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold yellow")
        log_tbl.add_column("Entry Date", width=12)
        log_tbl.add_column("Exit Date",  width=12)
        log_tbl.add_column("Stock",      width=14)
        log_tbl.add_column("Entry ₹",    justify="right", width=10)
        log_tbl.add_column("Exit ₹",     justify="right", width=10)
        log_tbl.add_column("Conviction", justify="right", width=11)
        log_tbl.add_column("Exit Type",  width=14)
        log_tbl.add_column("Net P&L (₹)",justify="right", width=12)
        log_tbl.add_column("Return (%)", justify="right", width=10)

        for t in sorted(forward_trades, key=lambda x: x.entry_time):
            c_col = "green" if t.net_pnl >= 0 else "red"
            c_sgn = "+" if t.net_pnl >= 0 else ""
            log_tbl.add_row(
                t.entry_time.strftime("%Y-%m-%d"),
                t.exit_time.strftime("%Y-%m-%d"),
                f"[bold]{t.symbol}[/]",
                f"₹{t.entry_price:,.2f}",
                f"₹{t.exit_price:,.2f}",
                f"[cyan]{t.index_prob:.1f}%[/]",
                t.exit_type,
                f"[{c_col}]{c_sgn}₹{t.net_pnl:,.2f}[/]",
                f"[{c_col}]{c_sgn}{t.net_pnl_pct:.2f}%[/]",
            )
        console.print(log_tbl)
        console.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", default="1d", help="Candle timeframe: 1d or 1h")
    parser.add_argument("--portfolio", action="store_true", help="Run institutional shared-book portfolio simulation")
    parser.add_argument("--max-per-master", type=int, default=3, help="Max open positions per master index")
    parser.add_argument("--max-open-trades", type=int, default=4, help="Max total open positions across entire book")
    parser.add_argument("--years", type=float, default=1.0, help="Number of years to backtest (e.g. 1, 3, 5, 10)")
    args = parser.parse_args()
    if args.portfolio:
        run_portfolio_test(interval=args.interval, max_positions_per_master=args.max_per_master, max_open_trades=args.max_open_trades, years=args.years)
    else:
        run_forward_test(interval=args.interval, years=args.years)
