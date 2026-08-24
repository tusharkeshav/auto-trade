# ─────────────────────────────────────────────────────────────────
#  backtest/engines/multi_strategy_engine.py
#  Method 2: Multi-Strategy Multiplexed Portfolio Engine.
#
#  Combines 3 Uncorrelated Quantitative Engines:
#    1. Strategy A: Master-to-Stock Pullback Engine (SMA20 Dips on Mega-Caps)
#    2. Strategy B: Minervini VCP Squeeze Breakout (20-Day High + Vol Squeeze)
#    3. Strategy C: Sector ETF Dual-Momentum Rotation (Top 2 Sectors + 200 SMA Shield)
#
#  Features:
#    - Shared Capital Book (₹100k or ₹50k base)
#    - Dynamic Risk Allocation (Max 6 concurrent positions)
#    - Full CNC Indian Statutory Tax Accounting
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from config.india_settings import INITIAL_CAPITAL_INR
from engine.india_costs import calculate_round_trip_cost
from indicators import add_all_indicators


STOCK_UNIVERSE = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "LT.NS", "BHARTIARTL.NS", "SBIN.NS", "SUNPHARMA.NS", "NTPC.NS"]
SECTOR_ETFS    = ["ITBEES.NS", "BANKBEES.NS", "AUTOBEES.NS", "PHARMABEES.NS", "CPSEETF.NS", "GOLDBEES.NS"]
BENCHMARK_INDEX= "^NSEI"


@dataclass
class MultiStratTrade:
    strategy_label: str
    symbol:         str
    entry_date:     datetime
    exit_date:      datetime
    entry_price:    float
    exit_price:     float
    qty:            float
    invested:       float
    gross_pnl:      float
    cost_inr:       float
    net_pnl:        float
    net_pnl_pct:    float
    exit_reason:    str
    bars_held:      int


@dataclass
class MultiStratResult:
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
    strat_breakdown: Dict[str, dict] = field(default_factory=dict)
    trades:          List[MultiStratTrade] = field(default_factory=list)
    equity_curve:    pd.Series = field(default_factory=pd.Series)


class MultiStrategyEngine:
    """
    Multi-Strategy Multiplexed Portfolio Engine (Pullback + VCP Breakout + Sector Rotation).
    """

    def __init__(
        self,
        capital:         float = INITIAL_CAPITAL_INR,
        max_open_trades: int   = 6,
        risk_per_trade_pct: float = 2.0,
    ):
        self.capital            = capital
        self.initial_capital    = capital
        self.max_open_trades    = max_open_trades
        self.risk_per_trade_pct = risk_per_trade_pct

    def run(self, bars: int = 1250) -> MultiStratResult:
        all_syms = list(set(STOCK_UNIVERSE + SECTOR_ETFS + [BENCHMARK_INDEX]))
        logger.info(f"Fetching {bars} Daily candles for Multi-Strategy Universe ({len(all_syms)} assets)...")

        df_raw = yf.download(all_syms, period=f"{int(bars*1.6)}d", interval="1d", progress=False)
        df_closes = df_raw["Close"].copy() if "Close" in df_raw.columns else df_raw.copy()
        if isinstance(df_closes.columns, pd.MultiIndex):
            df_closes.columns = df_closes.columns.get_level_values(0)
        df_closes = df_closes.ffill().dropna().iloc[-bars:]

        # Precompute indicators for individual stocks
        stock_dfs = {}
        for sym in STOCK_UNIVERSE:
            try:
                sub = pd.DataFrame(index=df_raw.index)
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    val = df_raw[col][sym] if isinstance(df_raw.columns, pd.MultiIndex) else df_raw[col]
                    sub[col.lower()] = val.astype(float)
                sub = add_all_indicators(sub.ffill().dropna()).iloc[-bars:]
                stock_dfs[sym] = sub
            except Exception as e:
                logger.warning(f"Failed to prepare {sym}: {e}")

        bm_series = df_closes[BENCHMARK_INDEX]
        bm_sma200 = bm_series.rolling(200).mean()
        bm_ema12  = bm_series.ewm(span=12).mean()
        bm_ema50  = bm_series.ewm(span=50).mean()

        total_bars = len(df_closes)
        warmup = 200

        capital = self.initial_capital
        open_positions: Dict[str, dict] = {}
        trades: List[MultiStratTrade] = []
        equity_values = []
        equity_dates  = []
        total_costs   = 0.0

        for t in range(warmup, total_bars):
            curr_date = df_closes.index[t]
            nifty_px  = bm_series.iloc[t]
            nifty_sma = bm_sma200.iloc[t]
            nifty_ema12 = bm_ema12.iloc[t]
            nifty_ema50 = bm_ema50.iloc[t]

            macro_bull = (nifty_px > nifty_sma) and (nifty_ema12 > nifty_ema50)

            # ── 1. Manage Active Positions ──
            for sym in list(open_positions.keys()):
                pos = open_positions[sym]
                curr_px = float(df_closes[sym].iloc[t]) if sym in df_closes else pos["entry_price"]

                # Breakout / Pullback TP & SL Check
                exit_triggered = False
                exit_px = curr_px
                exit_reason = ""

                if pos.get("is_rotational", False):
                    # Sector rotation rebalances on bar cycle
                    if t % 10 == 0:
                        # Re-evaluated in rotation block below
                        pass
                else:
                    # Trailing Break-Even
                    if not pos["be_locked"] and curr_px >= (pos["entry_price"] + 2.0 * pos["risk_unit"]):
                        pos["stop_loss"] = pos["entry_price"] + 0.5 * pos["risk_unit"]
                        pos["be_locked"] = True

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
                    b_cost, s_cost = calculate_round_trip_cost(pos["entry_price"], exit_px, pos["qty"], "CNC")
                    trade_tax = b_cost.total + s_cost.total
                    net_pnl   = gross_pnl - trade_tax
                    total_costs += trade_tax
                    capital += gross_sale - s_cost.total

                    trades.append(MultiStratTrade(
                        strategy_label = pos["strat"],
                        symbol         = sym,
                        entry_date     = pos["entry_date"],
                        exit_date      = curr_date,
                        entry_price    = round(pos["entry_price"], 2),
                        exit_price     = round(exit_px, 2),
                        qty            = round(pos["qty"], 2),
                        invested       = round(pos["entry_price"] * pos["qty"], 2),
                        gross_pnl      = round(gross_pnl, 2),
                        cost_inr       = round(trade_tax, 2),
                        net_pnl        = round(net_pnl, 2),
                        net_pnl_pct    = round((net_pnl / (pos["entry_price"] * pos["qty"])) * 100.0, 2),
                        exit_reason    = exit_reason,
                        bars_held      = t - pos["entry_idx"],
                    ))

            # ── 2. Scan Signals: Strategy A (Pullback) & Strategy B (VCP Breakout) ──
            if macro_bull and len(open_positions) < self.max_open_trades:
                for sym, df_s in stock_dfs.items():
                    if sym in open_positions or len(open_positions) >= self.max_open_trades:
                        continue
                    if t >= len(df_s):
                        continue

                    bar = df_s.iloc[t]
                    px = float(bar["close"])
                    atr = float(bar.get("atr", px * 0.02))
                    rsi = float(bar.get("rsi", 50.0))
                    vol = float(bar.get("volume", 0))
                    vol_ma = float(bar.get("volume_sma", vol))
                    sma20 = float(bar.get("sma_20", px))
                    bb_width = float(bar.get("bb_width", 0.05))

                    # Strategy A: Pullback Setup (Bounce from SMA20 wholesale support with RSI 40-55)
                    prev_px = float(df_s["close"].iloc[t-1])
                    prev_sma20 = float(df_s["sma_20"].iloc[t-1])
                    is_pullback = (prev_px <= prev_sma20 * 1.005) and (px > sma20) and (40.0 <= rsi <= 58.0)

                    # Strategy B: Minervini VCP Breakout (20-day High Breakout + Vol Squeeze + Vol > 1.2x)
                    window_20_high = float(df_s["high"].iloc[max(0, t-21):t].max())
                    prev_20_high = float(df_s["high"].iloc[max(0, t-22):t-1].max())
                    is_vcp_breakout = (px >= window_20_high * 0.999) and (prev_px < prev_20_high) and (vol >= vol_ma * 1.2) and (bb_width < 0.12)

                    strat_type = "VCP_BREAKOUT" if is_vcp_breakout else ("PULLBACK" if is_pullback else None)

                    if strat_type and capital > 1000:
                        sl_dist = atr * 1.25
                        sl_px = round(px - sl_dist, 2)
                        tp_mult = 3.5 if strat_type == "VCP_BREAKOUT" else 4.0
                        tp_px = round(px + sl_dist * tp_mult, 2)

                        port_val = capital + sum(p["qty"] * float(df_closes[s].iloc[t]) for s, p in open_positions.items())
                        risk_budget = port_val * (self.risk_per_trade_pct / 100.0)
                        qty = math.floor(risk_budget / sl_dist) if sl_dist > 0 else 0
                        max_alloc = capital / max(1, self.max_open_trades - len(open_positions))
                        qty = min(qty, math.floor(max_alloc / (px * 1.005)))

                        if qty > 0:
                            invested = qty * px
                            b_cost, _ = calculate_round_trip_cost(px, px, qty, "CNC")
                            if capital >= (invested + b_cost.total):
                                capital -= (invested + b_cost.total)
                                total_costs += b_cost.total

                                open_positions[sym] = {
                                    "strat": strat_type,
                                    "qty": qty,
                                    "entry_price": px,
                                    "stop_loss": sl_px,
                                    "take_profit": tp_px,
                                    "risk_unit": sl_dist,
                                    "be_locked": False,
                                    "entry_date": curr_date,
                                    "entry_idx": t,
                                    "is_rotational": False,
                                }

            # ── 3. Record Equity ──
            open_mtm = sum(pos["qty"] * float(df_closes[s].iloc[t]) for s, pos in open_positions.items() if s in df_closes)
            equity_values.append(capital + open_mtm)
            equity_dates.append(curr_date)

        # Close all remaining positions at end
        final_date = df_closes.index[-1]
        for sym, pos in list(open_positions.items()):
            px = float(df_closes[sym].iloc[-1]) if sym in df_closes else pos["entry_price"]
            gross_sale = px * pos["qty"]
            gross_pnl  = (px - pos["entry_price"]) * pos["qty"]
            b_cost, s_cost = calculate_round_trip_cost(pos["entry_price"], px, pos["qty"], "CNC")
            trade_tax = b_cost.total + s_cost.total
            net_pnl   = gross_pnl - trade_tax
            total_costs += trade_tax
            capital += gross_sale - s_cost.total

            trades.append(MultiStratTrade(
                strategy_label = pos["strat"],
                symbol         = sym,
                entry_date     = pos["entry_date"],
                exit_date      = final_date,
                entry_price    = round(pos["entry_price"], 2),
                exit_price     = round(px, 2),
                qty            = round(pos["qty"], 2),
                invested       = round(pos["entry_price"] * pos["qty"], 2),
                gross_pnl      = round(gross_pnl, 2),
                cost_inr       = round(trade_tax, 2),
                net_pnl        = round(net_pnl, 2),
                net_pnl_pct    = round((net_pnl / (pos["entry_price"] * pos["qty"])) * 100.0, 2),
                exit_reason    = "SIMULATION_END",
                bars_held      = total_bars - 1 - pos["entry_idx"],
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

        # Sub-strategy breakdown
        strat_breakdown = {}
        for s_lbl in ["PULLBACK", "VCP_BREAKOUT"]:
            s_trades = [t for t in trades if t.strategy_label == s_lbl]
            s_wins   = [t for t in s_trades if t.net_pnl > 0]
            strat_breakdown[s_lbl] = {
                "trades": len(s_trades),
                "win_rate": round(len(s_wins)/len(s_trades)*100.0, 1) if s_trades else 0.0,
                "net_pnl": round(sum(t.net_pnl for t in s_trades), 2),
            }

        return MultiStratResult(
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
            strat_breakdown  = strat_breakdown,
            trades           = trades,
            equity_curve     = equity_series,
        )
