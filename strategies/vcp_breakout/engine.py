# ─────────────────────────────────────────────────────────────────
#  strategies/vcp_breakout/engine.py
#  Standalone Minervini Volatility Contraction Pattern (VCP) Engine.
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
from indicators import add_all_indicators
from config.india_settings import INITIAL_CAPITAL_INR


DEFAULT_VCP_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "LT.NS",
    "BHARTIARTL.NS", "SBIN.NS", "SUNPHARMA.NS", "NTPC.NS"
]
BENCHMARK_SYMBOL = "^NSEI"


@dataclass
class VCPTrade:
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
class VCPResult:
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
    trades:          List[VCPTrade] = field(default_factory=list)
    equity_curve:    pd.Series = field(default_factory=pd.Series)


class VCPBreakoutEngine:
    def __init__(
        self,
        symbols:          Optional[List[str]] = None,
        benchmark_symbol: str   = BENCHMARK_SYMBOL,
        capital:          float = INITIAL_CAPITAL_INR,
        max_open_trades:  int   = 6,
        risk_per_trade_pct: float = 2.0,
        atr_sl_mult:      float = 1.25,
        atr_tp_mult:      float = 3.50,
        bb_width_threshold: float = 0.10,
        vol_ratio_threshold: float = 1.15,
    ):
        self.symbols             = symbols or DEFAULT_VCP_STOCKS
        self.benchmark_symbol    = benchmark_symbol
        self.initial_capital     = capital
        self.max_open_trades     = max_open_trades
        self.risk_per_trade_pct  = risk_per_trade_pct
        self.atr_sl_mult         = atr_sl_mult
        self.atr_tp_mult         = atr_tp_mult
        self.bb_width_threshold  = bb_width_threshold
        self.vol_ratio_threshold = vol_ratio_threshold

    def run(self, bars: int = 1250) -> VCPResult:
        all_syms = list(set(self.symbols + [self.benchmark_symbol]))
        df_raw = yf.download(all_syms, period=f"{int(bars*1.6)}d", interval="1d", progress=False)
        df_closes = df_raw["Close"].copy() if "Close" in df_raw.columns else df_raw.copy()
        if isinstance(df_closes.columns, pd.MultiIndex):
            df_closes.columns = df_closes.columns.get_level_values(0)
        df_closes = df_closes.ffill().dropna().iloc[-bars:]

        stock_dfs = {}
        for sym in self.symbols:
            try:
                sub = pd.DataFrame(index=df_raw.index)
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    val = df_raw[col][sym] if isinstance(df_raw.columns, pd.MultiIndex) else df_raw[col]
                    sub[col.lower()] = val.astype(float)
                sub = add_all_indicators(sub.ffill().dropna()).iloc[-bars:]
                stock_dfs[sym] = sub
            except Exception as e:
                logger.warning(f"Failed to prepare {sym}: {e}")

        bm_series = df_closes[self.benchmark_symbol]
        bm_sma200 = bm_series.rolling(200).mean()

        total_bars = len(df_closes)
        warmup = 200

        capital = self.initial_capital
        open_positions: Dict[str, dict] = {}
        trades: List[VCPTrade] = []
        equity_values, equity_dates = [], []
        total_costs = 0.0

        for t in range(warmup, total_bars):
            curr_date = df_closes.index[t]
            nifty_px  = bm_series.iloc[t]
            nifty_sma = bm_sma200.iloc[t]

            macro_bull = nifty_px > nifty_sma

            # Manage active positions
            for sym in list(open_positions.keys()):
                pos = open_positions[sym]
                curr_px = float(df_closes[sym].iloc[t]) if sym in df_closes else pos["entry_price"]

                # Trailing profit lock (+0.5R when profit reaches +2.0R)
                if not pos["be_locked"] and curr_px >= (pos["entry_price"] + 2.0 * pos["risk_unit"]):
                    pos["stop_loss"] = pos["entry_price"] + 0.5 * pos["risk_unit"]
                    pos["be_locked"] = True

                exit_triggered = False
                exit_px = curr_px
                exit_reason = ""

                if curr_px <= pos["stop_loss"]:
                    exit_triggered = True
                    exit_px = pos["stop_loss"]
                    exit_reason = "STOP_LOSS"
                elif curr_px >= pos["take_profit"]:
                    exit_triggered = True
                    exit_px = pos["take_profit"]
                    exit_reason = "TAKE_PROFIT"
                elif (t - pos["entry_idx"]) >= 45:
                    exit_triggered = True
                    exit_px = curr_px
                    exit_reason = "TIME_EXIT"

                if exit_triggered:
                    open_positions.pop(sym)
                    gross_sale = exit_px * pos["qty"]
                    gross_pnl  = (exit_px - pos["entry_price"]) * pos["qty"]
                    b_c, s_c   = calculate_round_trip_cost(pos["entry_price"], exit_px, pos["qty"], "CNC")
                    tax = b_c.total + s_c.total
                    net_pnl = gross_pnl - tax
                    total_costs += tax
                    capital += gross_sale - s_c.total

                    trades.append(VCPTrade(
                        symbol      = sym,
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
                        exit_reason = exit_reason,
                        bars_held   = t - pos["entry_idx"],
                    ))

            # Scan for Minervini VCP Breakouts
            if macro_bull and len(open_positions) < self.max_open_trades:
                for sym in self.symbols:
                    if sym in open_positions or len(open_positions) >= self.max_open_trades:
                        continue
                    df_s = stock_dfs.get(sym)
                    if df_s is None or t >= len(df_s):
                        continue

                    bar = df_s.iloc[t]
                    px = float(bar["close"])
                    sma200 = float(bar.get("sma_200", px))
                    sma50  = float(bar.get("sma_50", px))
                    atr    = float(bar.get("atr", px * 0.02))
                    vol_ratio = float(bar.get("volume_ratio", 1.0))
                    bb_w   = float(bar.get("bb_width", 0.05))

                    in_uptrend = (px > sma200) and (px > sma50)
                    w20_h = float(df_s["high"].iloc[max(0, t-21):t].max())
                    prev_px = float(df_s["close"].iloc[t-1])
                    prev_w20_h = float(df_s["high"].iloc[max(0, t-22):t-1].max())

                    is_vcp = in_uptrend and (px >= w20_h * 0.999) and (prev_px < prev_w20_h) and (vol_ratio >= self.vol_ratio_threshold) and (bb_w < self.bb_width_threshold)

                    if is_vcp and capital > 1000:
                        sl_dist = atr * self.atr_sl_mult
                        sl_px = round(px - sl_dist, 2)
                        tp_px = round(px + sl_dist * self.atr_tp_mult, 2)

                        port_val = capital + sum(p["qty"] * float(df_closes[s].iloc[t]) for s, p in open_positions.items() if s in df_closes)
                        risk_budget = port_val * (self.risk_per_trade_pct / 100.0)
                        qty = math.floor(risk_budget / sl_dist) if sl_dist > 0 else 0
                        max_alloc = capital / max(1, self.max_open_trades - len(open_positions))
                        qty = min(qty, math.floor(max_alloc / (px * 1.005)))

                        if qty > 0:
                            invested = qty * px
                            b_c, _ = calculate_round_trip_cost(px, px, qty, "CNC")
                            if capital >= (invested + b_c.total):
                                capital -= (invested + b_c.total)
                                total_costs += b_c.total
                                open_positions[sym] = {
                                    "qty": qty,
                                    "entry_price": px,
                                    "stop_loss": sl_px,
                                    "take_profit": tp_px,
                                    "risk_unit": sl_dist,
                                    "be_locked": False,
                                    "entry_date": curr_date,
                                    "entry_idx": t,
                                }

            open_mtm = sum(pos["qty"] * float(df_closes[s].iloc[t]) for s, pos in open_positions.items() if s in df_closes)
            equity_values.append(capital + open_mtm)
            equity_dates.append(curr_date)

        # Liquidate remaining
        final_date = df_closes.index[-1]
        for sym, pos in list(open_positions.items()):
            px = float(df_closes[sym].iloc[-1]) if sym in df_closes else pos["entry_price"]
            gross_sale = px * pos["qty"]
            gross_pnl  = (px - pos["entry_price"]) * pos["qty"]
            b_c, s_c   = calculate_round_trip_cost(pos["entry_price"], px, pos["qty"], "CNC")
            tax = b_c.total + s_c.total
            net_pnl = gross_pnl - tax
            total_costs += tax
            capital += gross_sale - s_c.total

            trades.append(VCPTrade(
                symbol      = sym,
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

        return VCPResult(
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
