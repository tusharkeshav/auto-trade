# ─────────────────────────────────────────────────────────────────
#  strategies/sector_rotation/engine.py
#  Sector ETF Dual-Momentum Rotation Engine with 200 SMA Shield.
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.india_costs import calculate_round_trip_cost
from config.india_settings import INITIAL_CAPITAL_INR


DEFAULT_SECTOR_ETFS = [
    "NIFTYBEES.NS",
    "BANKBEES.NS",
    "ITBEES.NS",
    "AUTOBEES.NS",
    "PHARMABEES.NS",
    "CPSEETF.NS",
]
SAFE_ASSET_SYMBOL = "GOLDBEES.NS"
BENCHMARK_SYMBOL  = "^NSEI"


@dataclass
class SectorTrade:
    symbol:       str
    entry_date:   datetime
    exit_date:    datetime
    entry_price:  float
    exit_price:   float
    qty:          float
    invested:     float
    gross_pnl:    float
    cost_inr:     float
    net_pnl:      float
    net_pnl_pct:  float
    exit_reason:  str
    bars_held:    int


@dataclass
class SectorResult:
    initial_capital: float
    final_capital:   float
    total_trades:    int
    winning_trades:  int
    losing_trades:   int
    win_rate:        float
    profit_factor:   float
    total_costs_inr: float
    net_pnl:         float
    net_pnl_pct:     float
    cagr_pct:        float
    max_drawdown_pct:float
    sharpe_ratio:    float
    sortino_ratio:   float
    trades:          List[SectorTrade] = field(default_factory=list)
    equity_curve:    pd.Series = field(default_factory=pd.Series)


class SectorRotationEngine:
    def __init__(
        self,
        symbols:           Optional[List[str]] = None,
        safe_symbol:       str   = SAFE_ASSET_SYMBOL,
        benchmark_symbol:  str   = BENCHMARK_SYMBOL,
        initial_capital:   float = INITIAL_CAPITAL_INR,
        momentum_window:   int   = 60,
        rebalance_interval:int   = 10,
        top_k:             int   = 2,
        use_macro_shield:  bool  = True,
        sma_filter_window: int   = 100,
    ):
        self.symbols            = symbols or DEFAULT_SECTOR_ETFS
        self.safe_symbol        = safe_symbol
        self.benchmark_symbol   = benchmark_symbol
        self.initial_capital    = initial_capital
        self.momentum_window    = momentum_window
        self.rebalance_interval = rebalance_interval
        self.top_k              = top_k
        self.use_macro_shield   = use_macro_shield
        self.sma_filter_window  = sma_filter_window

    def run(self, bars: int = 1250) -> SectorResult:
        all_syms = list(set(self.symbols + [self.safe_symbol, self.benchmark_symbol]))
        df_raw = yf.download(all_syms, period=f"{int(bars*1.6)}d", interval="1d", progress=False)["Close"]
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)
        df_close = df_raw.ffill().dropna().iloc[-bars:]

        total_bars = len(df_close)
        warmup = max(self.momentum_window, self.sma_filter_window, 200)

        bm_series = df_close[self.benchmark_symbol]
        bm_sma200 = bm_series.rolling(200).mean()

        etf_sma = {}
        for s in self.symbols:
            etf_sma[s] = df_close[s].rolling(self.sma_filter_window).mean()

        capital = self.initial_capital
        open_positions: Dict[str, dict] = {}
        trades: List[SectorTrade] = []
        equity_values, equity_dates = [], []
        total_costs = 0.0

        for t in range(warmup, total_bars):
            curr_date = df_close.index[t]
            curr_prices = {s: float(df_close[s].iloc[t]) for s in all_syms if s in df_close}

            macro_bull = True
            if self.use_macro_shield:
                macro_bull = curr_prices[self.benchmark_symbol] > bm_sma200.iloc[t]

            # Rebalance trigger
            if (t - warmup) % self.rebalance_interval == 0:
                target_allocations: Dict[str, float] = {}

                if not macro_bull:
                    target_allocations[self.safe_symbol] = 1.0
                else:
                    scores = []
                    for s in self.symbols:
                        px_now = curr_prices[s]
                        px_prev = float(df_close[s].iloc[t - self.momentum_window])
                        sma_val = etf_sma[s].iloc[t]
                        if px_now > sma_val and px_prev > 0:
                            ret = ((px_now - px_prev) / px_prev) * 100.0
                            scores.append((s, ret))
                    scores.sort(key=lambda x: x[1], reverse=True)
                    top_sectors = [s for s, r in scores[:self.top_k]]
                    if top_sectors:
                        w = 1.0 / len(top_sectors)
                        for s in top_sectors:
                            target_allocations[s] = w
                    else:
                        target_allocations[self.safe_symbol] = 1.0

                # Close positions no longer in target
                for s in list(open_positions.keys()):
                    if s not in target_allocations:
                        pos = open_positions.pop(s)
                        exit_px = curr_prices.get(s, pos["entry_price"])
                        gross_sale = exit_px * pos["qty"]
                        gross_pnl  = (exit_px - pos["entry_price"]) * pos["qty"]
                        b_c, s_c   = calculate_round_trip_cost(pos["entry_price"], exit_px, pos["qty"], "CNC")
                        tax = b_c.total + s_c.total
                        net_pnl = gross_pnl - tax
                        total_costs += tax
                        capital += gross_sale - s_c.total

                        trades.append(SectorTrade(
                            symbol      = s,
                            entry_date  = pos["entry_date"],
                            exit_date   = curr_date,
                            entry_price = round(pos["entry_price"], 2),
                            exit_price  = round(exit_px, 2),
                            qty         = round(pos["qty"], 2),
                            invested    = round(pos["entry_price"] * pos["qty"], 2),
                            gross_pnl   = round(gross_pnl, 2),
                            cost_inr    = round(tax, 2),
                            net_pnl     = round(net_pnl, 2),
                            net_pnl_pct = round((net_pnl / (pos["entry_price"] * pos["qty"])) * 100.0, 2),
                            exit_reason = "REBALANCE",
                            bars_held   = t - pos["entry_idx"],
                        ))

                # Allocate targets
                port_val = capital + sum(p["qty"] * curr_prices.get(s, p["entry_price"]) for s, p in open_positions.items())
                for s, weight in target_allocations.items():
                    if s not in open_positions and s in curr_prices:
                        px = curr_prices[s]
                        target_cash = port_val * weight
                        alloc_cash  = min(target_cash, capital * 0.98)
                        qty = math.floor(alloc_cash / (px * 1.002)) if px > 0 else 0
                        if qty > 0:
                            invested = qty * px
                            b_c, _ = calculate_round_trip_cost(px, px, qty, "CNC")
                            if capital >= (invested + b_c.total):
                                capital -= (invested + b_c.total)
                                total_costs += b_c.total
                                open_positions[s] = {
                                    "qty": qty,
                                    "entry_price": px,
                                    "entry_date": curr_date,
                                    "entry_idx": t,
                                }

            open_mtm = sum(pos["qty"] * curr_prices.get(s, pos["entry_price"]) for s, pos in open_positions.items())
            equity_values.append(capital + open_mtm)
            equity_dates.append(curr_date)

        # Final liquidation
        final_date = df_close.index[-1]
        for s, pos in list(open_positions.items()):
            px = float(df_close[s].iloc[-1]) if s in df_close else pos["entry_price"]
            gross_sale = px * pos["qty"]
            gross_pnl  = (px - pos["entry_price"]) * pos["qty"]
            b_c, s_c   = calculate_round_trip_cost(pos["entry_price"], px, pos["qty"], "CNC")
            tax = b_c.total + s_c.total
            net_pnl = gross_pnl - tax
            total_costs += tax
            capital += gross_sale - s_c.total

            trades.append(SectorTrade(
                symbol      = s,
                entry_date  = pos["entry_date"],
                exit_date   = final_date,
                entry_price = round(pos["entry_price"], 2),
                exit_price  = round(px, 2),
                qty         = round(pos["qty"], 2),
                invested    = round(pos["entry_price"] * pos["qty"], 2),
                gross_pnl   = round(gross_pnl, 2),
                cost_inr    = round(tax, 2),
                net_pnl     = round(net_pnl, 2),
                net_pnl_pct = round((net_pnl / (pos["entry_price"] * pos["qty"])) * 100.0, 2),
                exit_reason = "SIMULATION_END",
                bars_held   = total_bars - 1 - pos["entry_idx"],
            ))

        equity_series = pd.Series(equity_values, index=equity_dates)
        total_trades = len(trades)
        wins   = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]
        win_rate = (len(wins) / total_trades * 100.0) if total_trades else 0.0

        gross_profit = sum(t.net_pnl for t in wins)
        gross_loss   = abs(sum(t.net_pnl for t in losses))
        profit_factor= round(gross_profit / gross_loss, 2) if gross_loss else float("inf")

        net_pnl     = capital - self.initial_capital
        net_pnl_pct = (net_pnl / self.initial_capital) * 100.0

        years = len(equity_series) / 252.0 if len(equity_series) > 0 else 1.0
        cagr  = ((capital / self.initial_capital) ** (1.0 / years) - 1.0) * 100.0 if years > 0 and capital > 0 else 0.0

        peak   = equity_series.cummax()
        dd     = (peak - equity_series) / peak * 100.0
        max_dd = float(dd.max()) if not dd.empty else 0.0

        daily_returns = equity_series.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe  = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))
            neg_ret = daily_returns[daily_returns < 0]
            sortino = float((daily_returns.mean() / neg_ret.std()) * np.sqrt(252)) if len(neg_ret) > 1 and neg_ret.std() > 0 else sharpe
        else:
            sharpe, sortino = 0.0, 0.0

        return SectorResult(
            initial_capital  = self.initial_capital,
            final_capital    = round(capital, 2),
            total_trades     = total_trades,
            winning_trades   = len(wins),
            losing_trades    = len(losses),
            win_rate         = round(win_rate, 1),
            profit_factor    = profit_factor,
            total_costs_inr  = round(total_costs, 2),
            net_pnl          = round(net_pnl, 2),
            net_pnl_pct      = round(net_pnl_pct, 2),
            cagr_pct         = round(cagr, 2),
            max_drawdown_pct = round(max_dd, 2),
            sharpe_ratio     = round(sharpe, 2),
            sortino_ratio    = round(sortino, 2),
            trades           = trades,
            equity_curve     = equity_series,
        )
