# ─────────────────────────────────────────────────────────────────
#  run_exp_connors.py  —  Experiment 1: Larry Connors RSI(2)
#
#  Objective:
#    Test Larry Connors RSI(2) < 12.0 pullback in SMA200 uptrend
#    across Top 5 NIFTY Blue-Chips in pure Cash Delivery (CNC).
#
#  Usage:
#      python run_exp_connors.py --interval 1d --bars 500
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.india_settings import INITIAL_CAPITAL_INR, INDIA_MAX_RISK_PER_TRADE_PCT, INDIA_SIGNAL_THRESHOLD
from data.india.nse_client import NSEClient
from indicators import add_all_indicators
from probability.connors_scorer import ConnorsScorer
from engine.india_costs import calculate_round_trip_cost

console = Console()

TOP_5_STOCKS = ["LT.NS", "TCS.NS", "INFY.NS", "BHARTIARTL.NS", "SBIN.NS"]


@dataclass
class ConnorsTrade:
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
    probability: float
    bars_held:   int


@dataclass
class ConnorsResult:
    symbol:          str
    interval:        str
    period_start:    datetime
    period_end:      datetime
    candles:         int
    initial_capital: float
    trades:          list[ConnorsTrade] = field(default_factory=list)
    final_capital:   float = 0.0

    @property
    def total_trades(self) -> int: return len(self.trades)

    @property
    def winning_trades(self) -> list[ConnorsTrade]: return [t for t in self.trades if t.net_pnl > 0]

    @property
    def losing_trades(self) -> list[ConnorsTrade]: return [t for t in self.trades if t.net_pnl <= 0]

    @property
    def win_rate(self) -> float:
        return len(self.winning_trades) / self.total_trades * 100 if self.trades else 0.0

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


class ConnorsEngine:
    def __init__(self, symbol: str, interval: str = "1d", bars: int = 500, capital: float = 100_000.0):
        self.symbol   = symbol
        self.interval = interval
        self.bars     = bars
        self.capital  = capital
        self.client   = NSEClient()
        self.scorer   = ConnorsScorer(symbol=symbol, interval=interval)

    def run(self) -> ConnorsResult:
        logger.info(f"Fetching {self.bars} × {self.interval} candles for {self.symbol}...")
        df = add_all_indicators(self.client.get_ohlcv(self.symbol, self.interval, self.bars))
        WARMUP = min(200, len(df) // 4)

        result = ConnorsResult(
            symbol          = self.symbol,
            interval        = self.interval,
            period_start    = df.index[WARMUP],
            period_end      = df.index[-1],
            candles         = len(df) - WARMUP,
            initial_capital = self.capital,
            final_capital   = self.capital,
        )
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
                qty = min(qty, cash / entry) # 0% leverage
            if qty <= 0:
                i += 1
                continue

            trade = self._simulate_forward(df, signal, i, qty, entry, sl, tp)
            if trade:
                cash += trade.net_pnl
                result.trades.append(trade)
                i += trade.bars_held
                continue
            i += 1

        result.final_capital = round(cash, 2)
        return result

    def _simulate_forward(self, df: pd.DataFrame, signal, entry_idx: int, qty: float, entry: float, sl: float, tp: float) -> ConnorsTrade | None:
        for j in range(entry_idx + 1, len(df)):
            candle = df.iloc[j]
            high, low, close = float(candle["high"]), float(candle["low"]), float(candle["close"])
            bars = j - entry_idx

            if low <= sl:
                return self._make_trade(signal, df, entry_idx, j, entry, sl, "STOP_LOSS", qty, bars)
            if high >= tp:
                return self._make_trade(signal, df, entry_idx, j, entry, tp, "TAKE_PROFIT", qty, bars)
            # Connors fast exit: Close > SMA5
            sma5 = df.iloc[max(0, j - 4) : j + 1]["close"].mean()
            if close > sma5 and close > entry:
                return self._make_trade(signal, df, entry_idx, j, entry, close, "SMA5_EXIT", qty, bars)

        j = len(df) - 1
        return self._make_trade(signal, df, entry_idx, j, entry, float(df.iloc[j]["close"]), "TIMEOUT", qty, j - entry_idx)

    def _make_trade(self, signal, df, entry_idx, exit_idx, entry, exit_price, exit_type, qty, bars) -> ConnorsTrade:
        gross = (exit_price - entry) * qty
        buy_c, sell_c = calculate_round_trip_cost(entry, exit_price, qty, "CNC")
        cost  = buy_c.total + sell_c.total
        net   = gross - cost
        pct   = net / (entry * qty) * 100 if (entry * qty) > 0 else 0.0
        return ConnorsTrade(
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
            probability = signal.probability,
            bars_held   = bars,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", default="1d", help="Candle timeframe: 1d")
    parser.add_argument("--bars",     default=500,  type=int)
    args = parser.parse_args()

    console.print(f"\n[bold cyan]── EXPERIMENT 1: LARRY CONNORS RSI(2) < 12.0 PULLBACK ({args.interval}) ──[/]")
    console.print("[dim]Target: Top 5 Blue-Chips | CNC Delivery | 0% Leverage | Long-Only uptrend[/]\n")

    slice_cap = INITIAL_CAPITAL_INR / len(TOP_5_STOCKS)
    results = {}
    all_trades = []

    for sym in TOP_5_STOCKS:
        engine = ConnorsEngine(symbol=sym, interval=args.interval, bars=args.bars, capital=slice_cap)
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
