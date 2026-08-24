# ─────────────────────────────────────────────────────────────────
#  run_stock_portfolio_backtest.py  —  Multi-Stock Portfolio Backtest
#
#  Objective:
#    Validate "Momentum Crosses Mean" (Unified Cross) strategy across
#    a diversified portfolio of India's Top 5 Blue-Chip Cash Stocks:
#      1. RELIANCE.NS   (Energy / Conglomerate)
#      2. HDFCBANK.NS   (Banking / Finance)
#      3. ICICIBANK.NS  (Banking / Finance)
#      4. INFY.NS       (IT / Tech)
#      5. TATAMOTORS.NS (Auto / Industrial)
#
#  Execution Mechanics:
#    - Pure Cash Delivery (trade_type="CNC")
#    - Zero Leverage (Max shares capped by available cash balance)
#    - SEBI Compliance (Long-Only; short signals hard-blocked in CNC)
#    - Zero Brokerage (Zerodha/Dhan model; real STT/Exchange tax applied)
#
#  Usage:
#      python run_stock_portfolio_backtest.py --interval 1h --bars 1400
#      python run_stock_portfolio_backtest.py --interval 1d --bars 500
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.engines.unified_cross import UnifiedCrossBacktestEngine, UnifiedCrossResult, UnifiedCrossTrade
from config.india_settings import INITIAL_CAPITAL_INR, INDIA_SIGNAL_THRESHOLD

console = Console()

DEFAULT_PORTFOLIO_SYMBOLS = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "SBIN.NS",
]


@dataclass
class PortfolioSummary:
    interval:        str
    initial_capital: float
    results:         dict[str, UnifiedCrossResult] = field(default_factory=dict)
    all_trades:      list[UnifiedCrossTrade]       = field(default_factory=list)

    @property
    def total_trades(self) -> int: return len(self.all_trades)

    @property
    def winning_trades(self) -> list[UnifiedCrossTrade]: return [t for t in self.all_trades if t.pnl > 0]

    @property
    def losing_trades(self) -> list[UnifiedCrossTrade]: return [t for t in self.all_trades if t.pnl <= 0]

    @property
    def win_rate(self) -> float:
        return len(self.winning_trades) / self.total_trades * 100 if self.all_trades else 0.0

    @property
    def total_pnl(self) -> float:
        return sum(res.total_pnl for res in self.results.values())

    @property
    def total_pnl_pct(self) -> float:
        return self.total_pnl / self.initial_capital * 100

    @property
    def total_costs(self) -> float:
        return sum(res.total_costs_inr for res in self.results.values())

    @property
    def profit_factor(self) -> float:
        gp = sum(t.pnl for t in self.winning_trades)
        gl = abs(sum(t.pnl for t in self.losing_trades))
        return round(gp / gl, 2) if gl else float("inf")

    @property
    def max_drawdown_pct(self) -> float:
        if not self.all_trades:
            return 0.0
        # Sort all trades chronologically across portfolio
        sorted_trades = sorted(self.all_trades, key=lambda t: t.exit_time)
        capital = self.initial_capital
        peak, max_dd = capital, 0.0
        for t in sorted_trades:
            capital += t.pnl
            peak     = max(peak, capital)
            dd       = (peak - capital) / peak * 100
            max_dd   = max(max_dd, dd)
        return round(max_dd, 2)


def run_portfolio_simulation(symbols: list[str], interval: str, bars: int, capital: float, vix: float) -> PortfolioSummary:
    summary = PortfolioSummary(interval=interval, initial_capital=capital)

    # Allocate equal starting capital per stock slice in simulation
    slice_capital = capital / len(symbols)

    for sym in symbols:
        logger.info(f"Simulating {sym} ({interval}) with starting capital ₹{slice_capital:,.2f}...")
        engine = UnifiedCrossBacktestEngine(
            symbol      = sym,
            interval    = interval,
            bars        = bars,
            threshold   = INDIA_SIGNAL_THRESHOLD,
            atr_sl_mult = 1.0,
            atr_tp_mult = 2.5,
            capital     = slice_capital,
            vix         = vix,
            trade_type  = "CNC",  # Enforce pure CNC Delivery (zero brokerage, SEBI long-only cap)
        )
        res = engine.run()
        summary.results[sym] = res
        summary.all_trades.extend(res.trades)

    return summary


def print_portfolio_report(summary: PortfolioSummary) -> None:
    def sep(): console.print("[bold blue]" + "═" * 90 + "[/]")

    p_col = "green" if summary.total_pnl >= 0 else "red"
    p_sgn = "+" if summary.total_pnl >= 0 else ""
    pf_col = "green" if summary.profit_factor >= 1.5 else ("yellow" if summary.profit_factor >= 1.0 else "red")

    sep()
    console.print(f"[bold cyan]  MULTI-STOCK PORTFOLIO BACKTEST — TOP 5 NIFTY BLUE-CHIPS ({summary.interval})[/]")
    console.print("[dim]  Strategy: VWAP/EMA Pullback Cross + BB Reversal Cross (1:2.5 Asymmetric R)[/]")
    console.print("[dim]  Execution: Pure CNC Cash Delivery | 0% Brokerage | 0% Leverage | Long-Only[/]")
    sep()

    console.print(f"\n  [bold]── Aggregate Portfolio Performance ────────────────────────────────[/]")
    console.print(f"  Initial Capital      : [cyan]₹{summary.initial_capital:>12,.2f}[/]")
    console.print(f"  Final Capital        : [cyan]₹{summary.initial_capital + summary.total_pnl:>12,.2f}[/]")
    console.print(f"  Total Portfolio P&L  : [{p_col}]{p_sgn}₹{summary.total_pnl:>10,.2f}  ({p_sgn}{summary.total_pnl_pct:.2f}%)[/]")
    console.print(f"  Max Drawdown         : [red]{summary.max_drawdown_pct:.2f}%[/]  [dim](Portfolio level chronological peak-to-trough)[/]")
    console.print(f"  Total Taxes Paid     : [yellow]₹{summary.total_costs:,.2f}[/]  [dim](SEBI/Government STT & Exchange charges)[/]")

    console.print(f"\n  [bold]── Aggregate Trade Statistics ────────────────────────────────────[/]")
    console.print(f"  Total Trades         : [bold]{summary.total_trades}[/]  ({summary.total_trades / len(summary.results):.1f} avg per stock)")
    console.print(f"  Win Rate             : [yellow]{summary.win_rate:.1f}%[/]  ({len(summary.winning_trades)}W / {len(summary.losing_trades)}L)")
    console.print(f"  Profit Factor        : [{pf_col}]{summary.profit_factor:.2f}[/]  [dim](≥1.5 institutional grade)[/]")

    # Per-Stock Breakdown Table
    console.print(f"\n  [bold]── Individual Stock Breakdown ────────────────────────────────────[/]")
    tbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
    tbl.add_column("Symbol",        width=16)
    tbl.add_column("Trades",        justify="right", width=8)
    tbl.add_column("Win Rate",      justify="right", width=10)
    tbl.add_column("Profit Factor", justify="right", width=13)
    tbl.add_column("Taxes Paid (₹)",justify="right", width=14)
    tbl.add_column("Net P&L (₹)",   justify="right", width=14)
    tbl.add_column("Net Return (%)",justify="right", width=14)

    for sym, res in summary.results.items():
        s_col = "green" if res.total_pnl >= 0 else "red"
        s_sgn = "+" if res.total_pnl >= 0 else ""
        s_pfc = "green" if res.profit_factor >= 1.5 else ("yellow" if res.profit_factor >= 1.0 else "red")
        tbl.add_row(
            f"[bold]{sym}[/]",
            str(res.total_trades),
            f"{res.win_rate:.1f}%",
            f"[{s_pfc}]{res.profit_factor:.2f}[/]",
            f"₹{res.total_costs_inr:>9,.2f}",
            f"[{s_col}]{s_sgn}₹{res.total_pnl:>9,.2f}[/]",
            f"[{s_col}]{s_sgn}{res.total_pnl_pct:>6.2f}%[/]",
        )
    console.print(tbl)

    # Setup Breakdown across Portfolio
    console.print(f"\n  [bold]── Setup Breakdown (All Stocks Combined) ───────────────────────────[/]")
    for setup in ["PULLBACK", "BAND_BOUNCE", "DUAL_CONFIRM"]:
        st_trades = [t for t in summary.all_trades if t.setup_label == setup]
        cnt = len(st_trades)
        if cnt > 0:
            st_pnl = sum(t.pnl for t in st_trades)
            st_wr  = len([t for t in st_trades if t.pnl > 0]) / cnt * 100
            r_col  = "green" if st_pnl >= 0 else "red"
            r_sign = "+" if st_pnl >= 0 else ""
            console.print(f"  [cyan]{setup:<14}[/]  {cnt:>3} trades  |  WinRate: [yellow]{st_wr:>4.1f}%[/]  |  P&L: [{r_col}]{r_sign}₹{st_pnl:>9,.2f}[/]")
        else:
            console.print(f"  [dim]{setup:<14}    0 trades[/]")

    sep()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Stock Portfolio Backtest Runner")
    parser.add_argument("--interval", default="1h",       help="Candle timeframe: 1h or 1d")
    parser.add_argument("--bars",     default=1400,       type=int,   help="Number of historical candles")
    parser.add_argument("--capital",  default=INITIAL_CAPITAL_INR, type=float, help="Total portfolio starting capital INR (default: ₹5,00,000)")
    parser.add_argument("--vix",      default=16.0,       type=float, help="India VIX starting level")
    args = parser.parse_args()

    summary = run_portfolio_simulation(
        symbols  = DEFAULT_PORTFOLIO_SYMBOLS,
        interval = args.interval,
        bars     = args.bars,
        capital  = args.capital,
        vix      = args.vix,
    )
    print_portfolio_report(summary)


if __name__ == "__main__":
    main()
