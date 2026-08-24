# ─────────────────────────────────────────────────────────────────
#  run_exp_rotational.py  —  Experiment 2: Dual-Momentum Rotational
#
#  Objective:
#    Test Gary Antonacci's Rotational Momentum Model on Top 5 Stocks:
#      - Every 10 trading days (bi-weekly), rank stocks by 60-day return.
#      - Allocate 100% of cash equally across Top 2 strongest performers.
#      - If NIFTY50 < 200 SMA (bear market), sit in 100% Cash/LiquidBEES.
#      - Execution: Pure CNC Delivery | 0% Leverage | Real STT Tax.
#
#  Usage:
#      python run_exp_rotational.py --bars 500
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.india_settings import INITIAL_CAPITAL_INR
from data.india.nse_client import NSEClient
from engine.india_costs import calculate_round_trip_cost

console = Console()

TOP_5_STOCKS = ["LT.NS", "TCS.NS", "INFY.NS", "BHARTIARTL.NS", "SBIN.NS"]


@dataclass
class RotTrade:
    symbol:      str
    entry_time:  datetime
    exit_time:   datetime
    entry_price: float
    exit_price:  float
    qty:         float
    gross_pnl:   float
    cost_inr:    float
    net_pnl:     float
    net_pnl_pct: float
    bars_held:   int


@dataclass
class RotResult:
    trades:          list[RotTrade] = field(default_factory=list)
    initial_capital: float = INITIAL_CAPITAL_INR
    final_capital:   float = INITIAL_CAPITAL_INR

    @property
    def total_trades(self) -> int: return len(self.trades)

    @property
    def winning_trades(self) -> list[RotTrade]: return [t for t in self.trades if t.net_pnl > 0]

    @property
    def losing_trades(self) -> list[RotTrade]: return [t for t in self.trades if t.net_pnl <= 0]

    @property
    def win_rate(self) -> float: return len(self.winning_trades) / self.total_trades * 100 if self.trades else 0.0

    @property
    def total_pnl(self) -> float: return self.final_capital - self.initial_capital

    @property
    def total_pnl_pct(self) -> float: return self.total_pnl / self.initial_capital * 100

    @property
    def total_costs(self) -> float: return sum(t.cost_inr for t in self.trades)

    @property
    def profit_factor(self) -> float:
        gp = sum(t.net_pnl for t in self.winning_trades)
        gl = abs(sum(t.net_pnl for t in self.losing_trades))
        return round(gp / gl, 2) if gl else float("inf")


class RotationalEngine:
    def __init__(self, symbols: list[str], bars: int = 500, capital: float = INITIAL_CAPITAL_INR):
        self.symbols = symbols
        self.bars    = bars
        self.capital = capital
        self.client  = NSEClient()

    def run(self) -> RotResult:
        logger.info(f"Fetching {self.bars} × 1d candles for NIFTY50 & Top 5 Stocks...")
        df_nifty = self.client.get_ohlcv("NIFTY50", "1d", self.bars)
        df_nifty["sma_200"] = df_nifty["close"].rolling(200).mean()

        dfs = {}
        for sym in self.symbols:
            df = self.client.get_ohlcv(sym, "1d", self.bars)
            # 60-day rolling return (approx 3 months momentum)
            df["ret_60"] = df["close"].pct_change(60)
            dfs[sym] = df

        # Align timestamps across all stocks
        common_idx = df_nifty.index
        for df in dfs.values():
            common_idx = common_idx.intersection(df.index)

        df_nifty = df_nifty.loc[common_idx]
        for sym in self.symbols:
            dfs[sym] = dfs[sym].loc[common_idx]

        WARMUP = 200
        result = RotResult(initial_capital=self.capital, final_capital=self.capital)
        cash   = self.capital
        active_pos = {}  # sym -> (qty, entry_price, entry_str_time, entry_idx)

        i = WARMUP
        while i < len(common_idx):
            ts = common_idx[i]
            # Absolute momentum check: NIFTY > 200 SMA
            nifty_row = df_nifty.iloc[i]
            bull_market = not math.isnan(nifty_row["sma_200"]) and float(nifty_row["close"]) > float(nifty_row["sma_200"])

            # Rank stocks by 60-day return at index i
            scores = {}
            if bull_market:
                for sym, df in dfs.items():
                    val = float(df.iloc[i]["ret_60"])
                    if not math.isnan(val) and val > 0: # Must have positive momentum
                        scores[sym] = val

            # Top 2 performers
            top_2 = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:2]

            # 1. Sell any position no longer in Top 2 (or if bear market)
            for sym in list(active_pos.keys()):
                if sym not in top_2:
                    qty, entry_p, entry_ts, entry_idx = active_pos.pop(sym)
                    exit_p = float(dfs[sym].iloc[i]["close"])
                    gross  = (exit_p - entry_p) * qty
                    buy_c, sell_c = calculate_round_trip_cost(entry_p, exit_p, qty, "CNC")
                    cost   = buy_c.total + sell_c.total
                    net    = gross - cost
                    cash  += (entry_p * qty) + net
                    result.trades.append(RotTrade(
                        symbol      = sym,
                        entry_time  = entry_ts,
                        exit_time   = ts.to_pydatetime(),
                        entry_price = round(entry_p, 2),
                        exit_price  = round(exit_p, 2),
                        qty         = qty,
                        gross_pnl   = round(gross, 2),
                        cost_inr    = round(cost, 2),
                        net_pnl     = round(net, 2),
                        net_pnl_pct = round(net / (entry_p * qty) * 100, 4),
                        bars_held   = i - entry_idx,
                    ))

            # 2. Buy newly entering Top 2 stocks
            if top_2:
                target_alloc_per_stock = (cash + sum(dfs[s].iloc[i]["close"] * q for s, (q, _, _, _) in active_pos.items())) / len(top_2)
                for sym in top_2:
                    if sym not in active_pos:
                        price = float(dfs[sym].iloc[i]["close"])
                        alloc = min(cash, target_alloc_per_stock)
                        qty   = alloc / price if price > 0 else 0
                        if qty > 0 and (price * qty) <= cash:
                            cash -= (price * qty)
                            active_pos[sym] = (qty, price, ts.to_pydatetime(), i)

            # Rebalance every 10 trading days (~2 calendar weeks)
            i += 10

        # Close remaining at end
        i = len(common_idx) - 1
        ts = common_idx[i]
        for sym, (qty, entry_p, entry_ts, entry_idx) in active_pos.items():
            exit_p = float(dfs[sym].iloc[i]["close"])
            gross  = (exit_p - entry_p) * qty
            buy_c, sell_c = calculate_round_trip_cost(entry_p, exit_p, qty, "CNC")
            cost   = buy_c.total + sell_c.total
            net    = gross - cost
            cash  += (entry_p * qty) + net
            result.trades.append(RotTrade(
                symbol      = sym,
                entry_time  = entry_ts,
                exit_time   = ts.to_pydatetime(),
                entry_price = round(entry_p, 2),
                exit_price  = round(exit_p, 2),
                qty         = qty,
                gross_pnl   = round(gross, 2),
                cost_inr    = round(cost, 2),
                net_pnl     = round(net, 2),
                net_pnl_pct = round(net / (entry_p * qty) * 100, 4),
                bars_held   = i - entry_idx,
            ))

        result.final_capital = round(cash, 2)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", default=500, type=int)
    args = parser.parse_args()

    console.print("\n[bold cyan]── EXPERIMENT 2: DUAL-MOMENTUM ROTATIONAL ENGINE (1d) ──[/]")
    console.print("[dim]Target: Top 5 Blue-Chips | Bi-Weekly Rebalance | 0% Leverage | Real CNC Tax[/]\n")

    engine = RotationalEngine(symbols=TOP_5_STOCKS, bars=args.bars)
    r = engine.run()

    # Per-stock breakdown
    stk_trades = {}
    for sym in TOP_5_STOCKS:
        st = [t for t in r.trades if t.symbol == sym]
        stk_trades[sym] = st

    tbl = Table(box=box.DOUBLE_EDGE, show_header=True, header_style="bold cyan")
    tbl.add_column("Stock",       width=14)
    tbl.add_column("Trades",      justify="right", width=8)
    tbl.add_column("Win Rate",    justify="right", width=10)
    tbl.add_column("Profit Fact", justify="right", width=12)
    tbl.add_column("Taxes (₹)",   justify="right", width=12)
    tbl.add_column("Net P&L (₹)", justify="right", width=13)
    tbl.add_column("Avg Hold",    justify="right", width=10)

    for sym, st in stk_trades.items():
        w = len([t for t in st if t.net_pnl > 0])
        tr = len(st)
        wr = w / tr * 100 if tr else 0.0
        pnl = sum(t.net_pnl for t in st)
        cost = sum(t.cost_inr for t in st)
        gp = sum(t.net_pnl for t in st if t.net_pnl > 0)
        gl = abs(sum(t.net_pnl for t in st if t.net_pnl <= 0))
        pf = round(gp / gl, 2) if gl else float("inf")
        avg_h = sum(t.bars_held for t in st)/tr if tr else 0.0

        c_col = "green" if pnl >= 0 else "red"
        c_sgn = "+" if pnl >= 0 else ""
        f_col = "green" if pf >= 1.5 else ("yellow" if pf >= 1.0 else "red")
        tbl.add_row(
            f"[bold]{sym}[/]",
            str(tr),
            f"{wr:.1f}%",
            f"[{f_col}]{pf:.2f}[/]",
            f"₹{cost:>7,.2f}",
            f"[{c_col}]{c_sgn}₹{pnl:>8,.2f}[/]",
            f"{avg_h:.1f} d",
        )
    console.print(tbl)
    p_col = "green" if r.total_pnl >= 0 else "red"
    p_sgn = "+" if r.total_pnl >= 0 else ""
    console.print(f"\n[bold]PORTFOLIO SUMMARY[/]: Trades={r.total_trades} | Win Rate=[yellow]{r.win_rate:.1f}%[/] | "
                  f"PF=[cyan]{r.profit_factor}[/] | Taxes=₹{r.total_costs:,.2f} | Net P&L=[{p_col}]{p_sgn}₹{r.total_pnl:,.2f} ({p_sgn}{r.total_pnl_pct:.2f}%)[/]\n")


if __name__ == "__main__":
    main()
