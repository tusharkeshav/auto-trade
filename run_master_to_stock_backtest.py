# ─────────────────────────────────────────────────────────────────
#  run_master_to_stock_backtest.py  —  Master Index → Stock CNC Backtest
#
#  Objective:
#    Execute "Momentum Crosses Mean" using Rule 2:
#      - Generate clean signals from Spot Index (BANKNIFTY or NIFTY50)
#      - Execute trades on individual Cash Delivery stocks (CNC)
#
#  Stock Routing Map:
#    - HDFCBANK.NS, ICICIBANK.NS, SBIN.NS  ← Signal from BANKNIFTY Index
#    - RELIANCE.NS, INFY.NS                ← Signal from NIFTY50 Index
#
#  Usage:
#      python run_master_to_stock_backtest.py --interval 1h --bars 1400
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.engines.master_to_stock import MasterIndexToStockEngine, MasterStockResult, MasterStockTrade
from config.india_settings import INITIAL_CAPITAL_INR

console = Console()

STOCK_ROUTING_MAP = {
    "HDFCBANK.NS":  "BANKNIFTY",
    "ICICIBANK.NS": "BANKNIFTY",
    "SBIN.NS":      "BANKNIFTY",
    "RELIANCE.NS":  "NIFTY50",
    "INFY.NS":      "NIFTY50",
}


@dataclass
class MasterPortfolioSummary:
    interval:        str
    initial_capital: float
    results:         dict[str, MasterStockResult] = field(default_factory=dict)
    all_trades:      list[MasterStockTrade]       = field(default_factory=list)

    @property
    def total_trades(self) -> int: return len(self.all_trades)

    @property
    def winning_trades(self) -> list[MasterStockTrade]: return [t for t in self.all_trades if t.net_pnl > 0]

    @property
    def losing_trades(self) -> list[MasterStockTrade]: return [t for t in self.all_trades if t.net_pnl <= 0]

    @property
    def win_rate(self) -> float:
        return len(self.winning_trades) / self.total_trades * 100 if self.all_trades else 0.0

    @property
    def total_pnl(self) -> float: return sum(res.total_pnl for res in self.results.values())

    @property
    def total_pnl_pct(self) -> float: return self.total_pnl / self.initial_capital * 100

    @property
    def total_costs(self) -> float: return sum(res.total_costs_inr for res in self.results.values())

    @property
    def profit_factor(self) -> float:
        gp = sum(t.net_pnl for t in self.winning_trades)
        gl = abs(sum(t.net_pnl for t in self.losing_trades))
        return round(gp / gl, 2) if gl else float("inf")

    @property
    def max_drawdown_pct(self) -> float:
        if not self.all_trades:
            return 0.0
        sorted_trades = sorted(self.all_trades, key=lambda t: t.exit_time)
        capital = self.initial_capital
        peak, max_dd = capital, 0.0
        for t in sorted_trades:
            capital += t.net_pnl
            peak     = max(peak, capital)
            dd       = (peak - capital) / peak * 100
            max_dd   = max(max_dd, dd)
        return round(max_dd, 2)


def run_master_portfolio(routing_map: dict[str, str], interval: str, bars: int, capital: float, vix: float) -> MasterPortfolioSummary:
    summary = MasterPortfolioSummary(interval=interval, initial_capital=capital)
    slice_capital = capital / len(routing_map)

    for stk, master in routing_map.items():
        logger.info(f"Simulating {stk} (Signal: {master}, {interval}) with capital ₹{slice_capital:,.2f}...")
        engine = MasterIndexToStockEngine(
            stock_symbol = stk,
            master_index = master,
            interval     = interval,
            bars         = bars,
            capital      = slice_capital,
            vix          = vix,
        )
        res = engine.run()
        summary.results[stk] = res
        summary.all_trades.extend(res.trades)

    return summary


def print_master_report(summary: MasterPortfolioSummary) -> None:
    def sep(): console.print("[bold blue]" + "═" * 96 + "[/]")

    p_col = "green" if summary.total_pnl >= 0 else "red"
    p_sgn = "+" if summary.total_pnl >= 0 else ""
    pf_col = "green" if summary.profit_factor >= 1.5 else ("yellow" if summary.profit_factor >= 1.0 else "red")

    sep()
    console.print(f"[bold cyan]  MASTER INDEX → STOCK CNC PORTFOLIO BACKTEST ({summary.interval})[/]")
    console.print("[dim]  Rule 2: Generate clean signals on Spot Index | Execute on Stock CNC Delivery[/]")
    console.print("[dim]  Execution: 0% Leverage | 0% Brokerage | SEBI Long-Only | Real STT/Exchange Tax[/]")
    sep()

    console.print(f"\n  [bold]── Aggregate Portfolio Performance ────────────────────────────────[/]")
    console.print(f"  Initial Capital      : [cyan]₹{summary.initial_capital:>12,.2f}[/]")
    console.print(f"  Final Capital        : [cyan]₹{summary.initial_capital + summary.total_pnl:>12,.2f}[/]")
    console.print(f"  Total Portfolio P&L  : [{p_col}]{p_sgn}₹{summary.total_pnl:>10,.2f}  ({p_sgn}{summary.total_pnl_pct:.2f}%)[/]")
    console.print(f"  Max Drawdown         : [red]{summary.max_drawdown_pct:.2f}%[/]  [dim](Chronological peak-to-trough)[/]")
    console.print(f"  Total Taxes Paid     : [yellow]₹{summary.total_costs:,.2f}[/]  [dim](Real Government STT & Exchange tax)[/]")

    console.print(f"\n  [bold]── Aggregate Trade Statistics ────────────────────────────────────[/]")
    console.print(f"  Total Trades         : [bold]{summary.total_trades}[/]  ({summary.total_trades / len(summary.results):.1f} avg per stock)")
    console.print(f"  Win Rate             : [yellow]{summary.win_rate:.1f}%[/]  ({len(summary.winning_trades)}W / {len(summary.losing_trades)}L)")
    console.print(f"  Profit Factor        : [{pf_col}]{summary.profit_factor:.2f}[/]  [dim](≥1.5 institutional grade)[/]")

    console.print(f"\n  [bold]── Stock Routing & Performance Breakdown ─────────────────────────[/]")
    tbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
    tbl.add_column("Stock Symbol",  width=14)
    tbl.add_column("Master Index",  width=12)
    tbl.add_column("Trades",        justify="right", width=8)
    tbl.add_column("Win Rate",      justify="right", width=10)
    tbl.add_column("Profit Factor", justify="right", width=13)
    tbl.add_column("Taxes Paid (₹)",justify="right", width=14)
    tbl.add_column("Net P&L (₹)",   justify="right", width=14)
    tbl.add_column("Net Return (%)",justify="right", width=14)

    for stk, res in summary.results.items():
        s_col = "green" if res.total_pnl >= 0 else "red"
        s_sgn = "+" if res.total_pnl >= 0 else ""
        s_pfc = "green" if res.profit_factor >= 1.5 else ("yellow" if res.profit_factor >= 1.0 else "red")
        tbl.add_row(
            f"[bold]{stk}[/]",
            f"[cyan]{res.master_index}[/]",
            str(res.total_trades),
            f"{res.win_rate:.1f}%",
            f"[{s_pfc}]{res.profit_factor:.2f}[/]",
            f"₹{res.total_costs_inr:>9,.2f}",
            f"[{s_col}]{s_sgn}₹{res.total_pnl:>9,.2f}[/]",
            f"[{s_col}]{s_sgn}{res.total_pnl_pct:>6.2f}%[/]",
        )
    console.print(tbl)
    sep()


def main() -> None:
    parser = argparse.ArgumentParser(description="Master Index to Stock CNC Backtest Runner")
    parser.add_argument("--interval", default="1h",       help="Candle timeframe: 1h or 1d")
    parser.add_argument("--bars",     default=1400,       type=int,   help="Number of historical candles")
    parser.add_argument("--capital",  default=INITIAL_CAPITAL_INR, type=float, help="Total starting capital INR")
    parser.add_argument("--vix",      default=16.0,       type=float, help="India VIX starting level")
    args = parser.parse_args()

    summary = run_master_portfolio(
        routing_map = STOCK_ROUTING_MAP,
        interval    = args.interval,
        bars        = args.bars,
        capital     = args.capital,
        vix         = args.vix,
    )
    print_master_report(summary)


if __name__ == "__main__":
    main()
