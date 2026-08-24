# ─────────────────────────────────────────────────────────────────
#  run_exp_breakout.py  —  Experiment 3: Donchian 20-Day High Breakout
#
#  Objective:
#    Test 20-day high breakout in SMA200 uptrend across Top 5 Stocks
#    in pure Cash Delivery (CNC) with 1:3.5 Asymmetric target.
#
#  Usage:
#      python run_exp_breakout.py --bars 500
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.india_settings import INITIAL_CAPITAL_INR, INDIA_MAX_RISK_PER_TRADE_PCT
from data.india.nse_client import NSEClient
from indicators import add_all_indicators
from probability.breakout_scorer import BreakoutScorer
from engine.india_costs import calculate_round_trip_cost

console = Console()

TOP_5_STOCKS = ["LT.NS", "TCS.NS", "INFY.NS", "BHARTIARTL.NS", "SBIN.NS"]


@dataclass
class BrkTrade:
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
    exit_type:   str
    bars_held:   int


@dataclass
class BrkResult:
    trades:          list[BrkTrade] = field(default_factory=list)
    initial_capital: float = INITIAL_CAPITAL_INR
    final_capital:   float = INITIAL_CAPITAL_INR

    @property
    def total_trades(self) -> int: return len(self.trades)

    @property
    def winning_trades(self) -> list[BrkTrade]: return [t for t in self.trades if t.net_pnl > 0]

    @property
    def losing_trades(self) -> list[BrkTrade]: return [t for t in self.trades if t.net_pnl <= 0]

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


class BreakoutEngine:
    def __init__(self, symbol: str, bars: int = 500, capital: float = 100_000.0):
        self.symbol  = symbol
        self.bars    = bars
        self.capital = capital
        self.client  = NSEClient()
        self.scorer  = BreakoutScorer(symbol=symbol)

    def run(self) -> BrkResult:
        logger.info(f"Fetching {self.bars} × 1d candles for {self.symbol}...")
        df = add_all_indicators(self.client.get_ohlcv(self.symbol, "1d", self.bars))
        WARMUP = min(200, len(df) // 4)

        result = BrkResult(initial_capital=self.capital, final_capital=self.capital)
        cash = self.capital
        i = WARMUP
        while i < len(df) - 1:
            row = df.iloc[i]
            slice_df = df.iloc[: i + 1]
            signal = self.scorer.score(row, slice_df)
            if not signal.is_tradeable() or signal.direction == "SHORT":
                i += 1
                continue

            entry  = float(row["close"])
            sl_dist = signal.risk_amount
            sl      = signal.stop_loss
            tp      = signal.take_profit1

            risk_inr = cash * INDIA_MAX_RISK_PER_TRADE_PCT / 100
            qty      = risk_inr / sl_dist if sl_dist > 0 else 0
            if entry > 0:
                qty = min(qty, cash / entry)
            if qty <= 0:
                i += 1
                continue

            trade = self._simulate_forward(df, signal, i, qty, entry, sl, tp, sl_dist)
            if trade:
                cash += trade.net_pnl
                result.trades.append(trade)
                i += trade.bars_held
                continue
            i += 1

        result.final_capital = round(cash, 2)
        return result

    def _simulate_forward(self, df, signal, entry_idx, qty, entry, sl, tp, sl_dist) -> BrkTrade | None:
        for j in range(entry_idx + 1, len(df)):
            candle = df.iloc[j]
            high, low, close = float(candle["high"]), float(candle["low"]), float(candle["close"])
            bars = j - entry_idx

            if low <= sl:
                return self._make_trade(signal, df, entry_idx, j, entry, sl, "STOP_LOSS", qty, bars)
            if high >= tp:
                return self._make_trade(signal, df, entry_idx, j, entry, tp, "TAKE_PROFIT", qty, bars)

            # Donchian 10-day low exit or trailing stop
            if j >= 10:
                low_10 = df.iloc[j - 10 : j]["low"].min()
                if close < low_10 and close > entry:
                    return self._make_trade(signal, df, entry_idx, j, entry, close, "LOW_10_EXIT", qty, bars)

            # Trailing stop
            if high >= entry + sl_dist * 2.0 and sl < entry + sl_dist * 0.5:
                sl = round(entry + sl_dist * 0.5, 2)
            elif high >= entry + sl_dist * 1.0 and sl < entry:
                sl = round(entry, 2)

        j = len(df) - 1
        return self._make_trade(signal, df, entry_idx, j, entry, float(df.iloc[j]["close"]), "TIMEOUT", qty, j - entry_idx)

    def _make_trade(self, signal, df, entry_idx, exit_idx, entry, exit_price, exit_type, qty, bars) -> BrkTrade:
        gross = (exit_price - entry) * qty
        buy_c, sell_c = calculate_round_trip_cost(entry, exit_price, qty, "CNC")
        cost  = buy_c.total + sell_c.total
        net   = gross - cost
        pct   = net / (entry * qty) * 100 if (entry * qty) > 0 else 0.0
        return BrkTrade(
            symbol      = self.symbol,
            entry_time  = df.index[entry_idx].to_pydatetime(),
            exit_time   = df.index[exit_idx].to_pydatetime(),
            entry_price = round(entry, 2),
            exit_price  = round(exit_price, 2),
            qty         = qty,
            gross_pnl   = round(gross, 2),
            cost_inr    = round(cost, 2),
            net_pnl     = round(net, 2),
            net_pnl_pct = round(pct, 4),
            exit_type   = exit_type,
            bars_held   = bars,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", default=500, type=int)
    args = parser.parse_args()

    console.print("\n[bold cyan]── EXPERIMENT 3: DONCHIAN 20-DAY HIGH BREAKOUT ENGINE (1d) ──[/]")
    console.print("[dim]Target: Top 5 Blue-Chips | 1:3.5 Asymmetric R | 0% Leverage | Real CNC Tax[/]\n")

    slice_cap = INITIAL_CAPITAL_INR / len(TOP_5_STOCKS)
    results = {}
    all_trades = []

    for sym in TOP_5_STOCKS:
        engine = BreakoutEngine(symbol=sym, bars=args.bars, capital=slice_cap)
        res = engine.run()
        results[sym] = res
        all_trades.extend(res.trades)

    tot_pnl  = sum(r.total_pnl for r in results.values())
    tot_cost = sum(r.total_costs for r in results.values())
    tot_win  = len([t for t in all_trades if t.net_pnl > 0])
    tot_tr   = len(all_trades)
    win_rate = tot_win / tot_tr * 100 if tot_tr else 0.0
    gp = sum(t.net_pnl for t in all_trades if t.net_pnl > 0)
    gl = abs(sum(t.net_pnl for t in all_trades if t.net_pnl <= 0))
    pf = round(gp / gl, 2) if gl else float("inf")

    tbl = Table(box=box.DOUBLE_EDGE, show_header=True, header_style="bold cyan")
    tbl.add_column("Stock",       width=14)
    tbl.add_column("Trades",      justify="right", width=8)
    tbl.add_column("Win Rate",    justify="right", width=10)
    tbl.add_column("Profit Fact", justify="right", width=12)
    tbl.add_column("Taxes (₹)",   justify="right", width=12)
    tbl.add_column("Net P&L (₹)", justify="right", width=13)
    tbl.add_column("Return (%)",  justify="right", width=12)

    for sym, r in results.items():
        c_col = "green" if r.total_pnl >= 0 else "red"
        c_sgn = "+" if r.total_pnl >= 0 else ""
        f_col = "green" if r.profit_factor >= 1.5 else ("yellow" if r.profit_factor >= 1.0 else "red")
        tbl.add_row(
            f"[bold]{sym}[/]",
            str(r.total_trades),
            f"{r.win_rate:.1f}%",
            f"[{f_col}]{r.profit_factor:.2f}[/]",
            f"₹{r.total_costs:>7,.2f}",
            f"[{c_col}]{c_sgn}₹{r.total_pnl:>8,.2f}[/]",
            f"[{c_col}]{c_sgn}{r.total_pnl_pct:>5.2f}%[/]",
        )
    console.print(tbl)
    p_col = "green" if tot_pnl >= 0 else "red"
    p_sgn = "+" if tot_pnl >= 0 else ""
    console.print(f"\n[bold]PORTFOLIO SUMMARY[/]: Trades={tot_tr} | Win Rate=[yellow]{win_rate:.1f}%[/] | "
                  f"PF=[cyan]{pf}[/] | Taxes=₹{tot_cost:,.2f} | Net P&L=[{p_col}]{p_sgn}₹{tot_pnl:,.2f} ({p_sgn}{tot_pnl/INITIAL_CAPITAL_INR*100:.2f}%)[/]\n")


if __name__ == "__main__":
    main()
