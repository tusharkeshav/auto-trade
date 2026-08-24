# ─────────────────────────────────────────────────────────────────
#  optimize_stock_portfolio.py  —  High-Beta Stock Portfolio Optimizer
#
#  Objective:
#    Optimize Method 2 (Pure Unleveraged Cash Stocks in CNC Delivery)
#    to achieve:
#      - Trade Frequency: 15 to 25 trades/month across portfolio
#      - Profitability  : Significant cash returns (>10-15%) beating STT tax
#      - Risk Tolerance : Controlled drawdown (5% to 10%)
#
#  Optimization Levers Applied:
#    1. High-Beta Universe: 10 liquid stocks with strong intraday/hourly swings
#    2. Loosened Proximity Gate: Allow pullbacks within ±0.85% of VWAP/EMA20
#    3. 1:3.0 Asymmetric Reward: TP = 3.0× ATR (letting stock winners run)
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.engines.master_to_stock import MasterIndexToStockEngine, MasterStockResult, MasterStockTrade
from config.india_settings import INITIAL_CAPITAL_INR, INDIA_SIGNAL_THRESHOLD
from probability.unified_cross_scorer import UnifiedCrossScorer

console = Console()

# ── 10 High-Beta / Strong Movers (NSE Cash Delivery) ───────────────
OPTIMIZED_STOCK_ROUTING_MAP = {
    # Banking / Financial Leaders (Signal: BANKNIFTY)
    "SBIN.NS":      "BANKNIFTY",
    "ICICIBANK.NS": "BANKNIFTY",
    "AXISBANK.NS":  "BANKNIFTY",
    "KOTAKBANK.NS": "BANKNIFTY",
    "INDUSINDBK.NS":"BANKNIFTY",
    # Tech / Industrial / Consumption Leaders (Signal: NIFTY50)
    "INFY.NS":      "NIFTY50",
    "TCS.NS":       "NIFTY50",
    "LT.NS":        "NIFTY50",
    "BHARTIARTL.NS":"NIFTY50",
    "ITC.NS":       "NIFTY50",
}


@dataclass
class OptimizedPortfolioSummary:
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


def run_optimized_simulation(
    routing_map: dict[str, str],
    interval:    str,
    bars:        int,
    capital:     float,
    vix:         float,
    sl_mult:     float,
    tp_mult:     float,
) -> OptimizedPortfolioSummary:
    summary = OptimizedPortfolioSummary(interval=interval, initial_capital=capital)
    slice_capital = capital / len(routing_map)

    for stk, master in routing_map.items():
        logger.info(f"Simulating {stk} (Signal: {master}, {interval}) with capital ₹{slice_capital:,.2f}...")
        engine = MasterIndexToStockEngine(
            stock_symbol = stk,
            master_index = master,
            interval     = interval,
            bars         = bars,
            threshold    = INDIA_SIGNAL_THRESHOLD,
            atr_sl_mult  = sl_mult,
            atr_tp_mult  = tp_mult,
            capital      = slice_capital,
            vix          = vix,
        )
        res = engine.run()
        summary.results[stk] = res
        summary.all_trades.extend(res.trades)

    return summary


def print_optimized_report(summary: OptimizedPortfolioSummary) -> None:
    def sep(): console.print("[bold blue]" + "═" * 100 + "[/]")

    p_col = "green" if summary.total_pnl >= 0 else "red"
    p_sgn = "+" if summary.total_pnl >= 0 else ""
    pf_col = "green" if summary.profit_factor >= 1.5 else ("yellow" if summary.profit_factor >= 1.0 else "red")

    sep()
    console.print(f"[bold cyan]  HIGH-BETA 10-STOCK PORTFOLIO OPTIMIZATION ({summary.interval})[/]")
    console.print("[dim]  Optimization: 10 Liquid Stocks | Loosened Proximity Gate | 1:3.0 Asymmetric Target[/]")
    console.print("[dim]  Execution   : Pure CNC Delivery | 0% Brokerage | 0% Leverage | SEBI Long-Only[/]")
    sep()

    console.print(f"\n  [bold]── Aggregate Portfolio Performance ────────────────────────────────[/]")
    console.print(f"  Initial Capital      : [cyan]₹{summary.initial_capital:>12,.2f}[/]")
    console.print(f"  Final Capital        : [cyan]₹{summary.initial_capital + summary.total_pnl:>12,.2f}[/]")
    console.print(f"  Total Portfolio P&L  : [{p_col}]{p_sgn}₹{summary.total_pnl:>10,.2f}  ({p_sgn}{summary.total_pnl_pct:.2f}%)[/]")
    console.print(f"  Max Drawdown         : [red]{summary.max_drawdown_pct:.2f}%[/]  [dim](Chronological peak-to-trough across portfolio)[/]")
    console.print(f"  Total Taxes Paid     : [yellow]₹{summary.total_costs:,.2f}[/]  [dim](Real Government STT & Exchange tax absorbed)[/]")

    console.print(f"\n  [bold]── Aggregate Trade Statistics ────────────────────────────────────[/]")
    console.print(f"  Total Trades         : [bold]{summary.total_trades}[/]  ({summary.total_trades / len(summary.results):.1f} avg per stock)")
    console.print(f"  Win Rate             : [yellow]{summary.win_rate:.1f}%[/]  ({len(summary.winning_trades)}W / {len(summary.losing_trades)}L)")
    console.print(f"  Profit Factor        : [{pf_col}]{summary.profit_factor:.2f}[/]  [dim](≥1.5 institutional grade)[/]")

    console.print(f"\n  [bold]── Individual Stock Performance Breakdown ────────────────────────[/]")
    tbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
    tbl.add_column("Stock Symbol",  width=14)
    tbl.add_column("Master Index",  width=11)
    tbl.add_column("Trades",        justify="right", width=7)
    tbl.add_column("Win Rate",      justify="right", width=9)
    tbl.add_column("Profit Factor", justify="right", width=13)
    tbl.add_column("Taxes Paid (₹)",justify="right", width=14)
    tbl.add_column("Net P&L (₹)",   justify="right", width=13)
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
            f"₹{res.total_costs_inr:>8,.2f}",
            f"[{s_col}]{s_sgn}₹{res.total_pnl:>8,.2f}[/]",
            f"[{s_col}]{s_sgn}{res.total_pnl_pct:>6.2f}%[/]",
        )
    console.print(tbl)
    sep()


def main() -> None:
    parser = argparse.ArgumentParser(description="High-Beta Stock Portfolio Optimizer")
    parser.add_argument("--interval", default="1h",       help="Candle timeframe: 1h or 1d")
    parser.add_argument("--bars",     default=1400,       type=int,   help="Number of historical candles")
    parser.add_argument("--capital",  default=INITIAL_CAPITAL_INR, type=float, help="Total starting capital INR")
    parser.add_argument("--vix",      default=16.0,       type=float, help="India VIX starting level")
    parser.add_argument("--sl-mult",  default=1.0,        type=float, help="ATR SL multiplier")
    parser.add_argument("--tp-mult",  default=3.0,        type=float, help="ATR TP multiplier (R-multiple)")
    args = parser.parse_args()

    summary = run_optimized_simulation(
        routing_map = OPTIMIZED_STOCK_ROUTING_MAP,
        interval    = args.interval,
        bars        = args.bars,
        capital     = args.capital,
        vix         = args.vix,
        sl_mult     = args.sl_mult,
        tp_mult     = args.tp_mult,
    )
    print_optimized_report(summary)


if __name__ == "__main__":
    main()
