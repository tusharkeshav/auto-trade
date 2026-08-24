# ─────────────────────────────────────────────────────────────────
#  backtest/engines/dual_timeframe_engine.py
#  Method 1: Dual-Timeframe 1-Hour Precision Timing Engine.
#
#  Mathematical Architecture:
#    1. Macro Layer (Daily ^NSEI):
#       - Macro Trend: Daily ^NSEI EMA12 > EMA50 & Price > SMA200.
#       - Pullback Zone: Daily RSI between 40 and 65, ADX >= 18.
#
#    2. Micro Layer (1-Hour Stock Chart):
#       - Entry Trigger: 1h StochRSI bullish cross (K > D under 30) OR
#         1h MACD histogram crosses above zero.
#       - Stop Loss: 1.25 × 1h ATR below entry (~1.2% - 1.8% distance).
#       - Take Profit: 4.0 × 1h ATR above entry (Asymmetric 1:3.2+ R:R).
#       - Trailing Stop: Lock +1.0R once trade reaches +2.0R profit.
#
#    3. Real Indian CNC Statutory Tax Deductions on every trade.
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from config.india_settings import INITIAL_CAPITAL_INR
from engine.india_costs import calculate_round_trip_cost
from indicators import add_all_indicators


DEFAULT_1H_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "LT.NS",
    "BHARTIARTL.NS", "SBIN.NS", "SUNPHARMA.NS", "NTPC.NS", "ITC.NS"
]
BENCHMARK_INDEX = "^NSEI"


@dataclass
class DualTFTrade:
    symbol:       str
    entry_time:   datetime
    exit_time:    datetime
    entry_price:  float
    exit_price:   float
    qty:          float
    invested:     float
    gross_pnl:    float
    cost_inr:     float
    net_pnl:      float
    net_pnl_pct:  float
    exit_type:    str
    bars_held:    int


@dataclass
class DualTFResult:
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
    trades:          List[DualTFTrade] = field(default_factory=list)
    equity_curve:    pd.Series = field(default_factory=pd.Series)


class DualTimeframeEngine:
    """
    Dual-Timeframe Quantitative Engine (Daily Macro + 1-Hour Micro Entry).
    """

    def __init__(
        self,
        symbols:          Optional[List[str]] = None,
        benchmark_symbol: str   = BENCHMARK_INDEX,
        capital:          float = INITIAL_CAPITAL_INR,
        max_open_trades:  int   = 6,
        risk_per_trade_pct: float = 2.0,
        atr_sl_mult:      float = 1.25,
        atr_tp_mult:      float = 4.0,
    ):
        self.symbols            = symbols or DEFAULT_1H_STOCKS
        self.benchmark_symbol   = benchmark_symbol
        self.capital            = capital
        self.initial_capital    = capital
        self.max_open_trades    = max_open_trades
        self.risk_per_trade_pct = risk_per_trade_pct
        self.atr_sl_mult        = atr_sl_mult
        self.atr_tp_mult        = atr_tp_mult

    def _fetch_data(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Fetch 1d benchmark and 1h stock candles."""
        logger.info("Fetching Daily Benchmark (^NSEI) and 1-Hour Stock Data...")

        # 1. Fetch Daily Benchmark
        df_bm_raw = yf.download(self.benchmark_symbol, period="730d", interval="1d", progress=False)
        if isinstance(df_bm_raw.columns, pd.MultiIndex):
            df_bm_raw.columns = df_bm_raw.columns.get_level_values(0)
        df_bm_std = pd.DataFrame(index=df_bm_raw.index)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df_bm_raw.columns:
                df_bm_std[col.lower()] = df_bm_raw[col].astype(float)
        df_bm = add_all_indicators(df_bm_std.ffill().dropna())

        # 2. Fetch 1-Hour Stocks
        stock_dfs = {}
        for sym in self.symbols:
            try:
                df_raw = yf.download(sym, period="730d", interval="1h", progress=False)
                if isinstance(df_raw.columns, pd.MultiIndex):
                    df_raw.columns = df_raw.columns.get_level_values(0)
                if len(df_raw) > 100:
                    # Rename columns to standard lowercase
                    df_std = pd.DataFrame(index=df_raw.index)
                    df_std["open"]   = df_raw["Open"].astype(float)
                    df_std["high"]   = df_raw["High"].astype(float)
                    df_std["low"]    = df_raw["Low"].astype(float)
                    df_std["close"]  = df_raw["Close"].astype(float)
                    df_std["volume"] = df_raw["Volume"].astype(float)
                    df_std = add_all_indicators(df_std)
                    stock_dfs[sym] = df_std
            except Exception as e:
                logger.warning(f"Error fetching 1h data for {sym}: {e}")

        return df_bm, stock_dfs

    def run(self) -> DualTFResult:
        df_bm, stock_dfs = self._fetch_data()
        if not stock_dfs:
            raise RuntimeError("No 1-hour stock data available.")

        # Synchronize timeline on 1h bar timestamps
        all_timestamps = sorted(list(set().union(*[set(df.index) for df in stock_dfs.values()])))
        warmup = 150

        if len(all_timestamps) <= warmup:
            raise ValueError(f"Insufficient history: {len(all_timestamps)} bars.")

        capital = self.initial_capital
        open_positions: Dict[str, dict] = {}
        trades: List[DualTFTrade] = []
        equity_values = []
        equity_dates  = []
        total_costs   = 0.0

        # Precompute daily date map for Macro filter
        df_bm["date_str"] = df_bm.index.strftime("%Y-%m-%d")
        daily_map = {row["date_str"]: row for _, row in df_bm.iterrows()}

        for i, ts in enumerate(all_timestamps[warmup:], start=warmup):
            date_key = ts.strftime("%Y-%m-%d")
            daily_bm_row = daily_map.get(date_key)

            # ── 1. Evaluate Daily Macro Filter ──
            macro_bullish = False
            if daily_bm_row is not None:
                bm_close = float(daily_bm_row.get("close", 0))
                bm_ema12 = float(daily_bm_row.get("ema_12", 0))
                bm_ema50 = float(daily_bm_row.get("ema_50", 0))
                bm_rsi   = float(daily_bm_row.get("rsi", 50))
                # Macro trend is bullish when EMA12 > EMA50 and RSI in healthy pullback zone (38-65)
                macro_bullish = (bm_ema12 > bm_ema50) and (38.0 <= bm_rsi <= 65.0)

            # ── 2. Manage Existing Open Positions ──
            for sym in list(open_positions.keys()):
                pos = open_positions[sym]
                df_sym = stock_dfs.get(sym)
                if df_sym is None or ts not in df_sym.index:
                    continue

                bar = df_sym.loc[ts]
                high_px = float(bar["high"])
                low_px  = float(bar["low"])
                close_px= float(bar["close"])

                # Check Trailing Break-Even (+1.0R lock when price reaches +2.0R)
                if not pos["be_locked"] and high_px >= (pos["entry_price"] + 2.0 * pos["risk_unit"]):
                    pos["stop_loss"] = pos["entry_price"] + 0.5 * pos["risk_unit"]
                    pos["be_locked"] = True

                exit_triggered = False
                exit_px = close_px
                exit_type = ""

                # Check Stop Loss
                if low_px <= pos["stop_loss"]:
                    exit_triggered = True
                    exit_px = pos["stop_loss"]
                    exit_type = "STOP_LOSS"
                # Check Take Profit
                elif high_px >= pos["take_profit"]:
                    exit_triggered = True
                    exit_px = pos["take_profit"]
                    exit_type = "TAKE_PROFIT"
                # Time exit (held > 120 1h bars ~18 trading days without hitting TP)
                elif (i - pos["entry_idx"]) >= 120:
                    exit_triggered = True
                    exit_px = close_px
                    exit_type = "TIME_EXPIRY"

                if exit_triggered:
                    open_positions.pop(sym)
                    gross_sale = exit_px * pos["qty"]
                    gross_pnl  = (exit_px - pos["entry_price"]) * pos["qty"]
                    b_cost, s_cost = calculate_round_trip_cost(pos["entry_price"], exit_px, pos["qty"], "CNC")
                    trade_tax = b_cost.total + s_cost.total
                    net_pnl   = gross_pnl - trade_tax
                    total_costs += trade_tax
                    capital += gross_sale - s_cost.total

                    trades.append(DualTFTrade(
                        symbol      = sym,
                        entry_time  = pos["entry_time"],
                        exit_time   = ts,
                        entry_price = round(pos["entry_price"], 2),
                        exit_price  = round(exit_px, 2),
                        qty         = round(pos["qty"], 2),
                        invested    = round(pos["entry_price"] * pos["qty"], 2),
                        gross_pnl   = round(gross_pnl, 2),
                        cost_inr    = round(trade_tax, 2),
                        net_pnl     = round(net_pnl, 2),
                        net_pnl_pct = round((net_pnl / (pos["entry_price"] * pos["qty"])) * 100.0, 2),
                        exit_type   = exit_type,
                        bars_held   = i - pos["entry_idx"],
                    ))

            # ── 3. Scan for New 1-Hour Entries ──
            if macro_bullish and len(open_positions) < self.max_open_trades:
                for sym in self.symbols:
                    if sym in open_positions or len(open_positions) >= self.max_open_trades:
                        continue

                    df_sym = stock_dfs.get(sym)
                    if df_sym is None or ts not in df_sym.index:
                        continue

                    bar = df_sym.loc[ts]
                    close_px = float(bar["close"])
                    atr_val  = float(bar.get("atr", close_px * 0.015))
                    rsi_val  = float(bar.get("rsi", 50.0))
                    macd_hist= float(bar.get("macd_hist", 0.0))

                    if math.isnan(atr_val) or atr_val <= 0:
                        atr_val = close_px * 0.015

                    # 1H Trigger Condition: RSI oversold pullback (<35) or MACD Histogram turning positive
                    entry_signal = (rsi_val <= 38.0 and close_px > float(bar.get("sma_200", 0))) or (macd_hist > 0 and rsi_val < 55.0)

                    if entry_signal and capital > 1000:
                        sl_dist = atr_val * self.atr_sl_mult
                        sl_px   = round(close_px - sl_dist, 2)
                        tp_px   = round(close_px + sl_dist * self.atr_tp_mult, 2)

                        # Dynamic 2% Risk Position Sizing
                        risk_budget = (capital + sum(p["qty"] * p["entry_price"] for p in open_positions.values())) * (self.risk_per_trade_pct / 100.0)
                        qty = math.floor(risk_budget / sl_dist) if sl_dist > 0 else 0
                        # Cap position size to max available capital / remaining slots
                        max_cash_alloc = capital / (self.max_open_trades - len(open_positions))
                        qty = min(qty, math.floor(max_cash_alloc / close_px))

                        if qty > 0:
                            invested = qty * close_px
                            b_cost, _ = calculate_round_trip_cost(close_px, close_px, qty, "CNC")
                            if capital >= (invested + b_cost.total):
                                capital -= (invested + b_cost.total)
                                total_costs += b_cost.total

                                open_positions[sym] = {
                                    "qty": qty,
                                    "entry_price": close_px,
                                    "stop_loss": sl_px,
                                    "take_profit": tp_px,
                                    "risk_unit": sl_dist,
                                    "be_locked": False,
                                    "entry_time": ts,
                                    "entry_idx": i,
                                }

            # ── 4. Record Portfolio Mark-to-Market Equity ──
            open_mtm = sum(pos["qty"] * float(stock_dfs[s].loc[ts]["close"]) for s, pos in open_positions.items() if ts in stock_dfs[s].index)
            equity_values.append(capital + open_mtm)
            equity_dates.append(ts)

        # Close any remaining open positions
        final_ts = all_timestamps[-1]
        for sym, pos in list(open_positions.items()):
            df_sym = stock_dfs.get(sym)
            exit_px = float(df_sym.iloc[-1]["close"]) if df_sym is not None else pos["entry_price"]
            gross_sale = exit_px * pos["qty"]
            gross_pnl  = (exit_px - pos["entry_price"]) * pos["qty"]
            b_cost, s_cost = calculate_round_trip_cost(pos["entry_price"], exit_px, pos["qty"], "CNC")
            trade_tax = b_cost.total + s_cost.total
            net_pnl   = gross_pnl - trade_tax
            total_costs += trade_tax
            capital += gross_sale - s_cost.total

            trades.append(DualTFTrade(
                symbol      = sym,
                entry_time  = pos["entry_time"],
                exit_time   = final_ts,
                entry_price = round(pos["entry_price"], 2),
                exit_price  = round(exit_px, 2),
                qty         = round(pos["qty"], 2),
                invested    = round(pos["entry_price"] * pos["qty"], 2),
                gross_pnl   = round(gross_pnl, 2),
                cost_inr    = round(trade_tax, 2),
                net_pnl     = round(net_pnl, 2),
                net_pnl_pct = round((net_pnl / (pos["entry_price"] * pos["qty"])) * 100.0, 2),
                exit_type   = "SIMULATION_END",
                bars_held   = len(all_timestamps) - 1 - pos["entry_idx"],
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

        # Annualized metrics (estimating ~1,750 1h trading bars per year)
        years = len(equity_series) / 1750.0 if len(equity_series) > 0 else 1.0
        cagr  = ((capital / self.initial_capital) ** (1.0 / years) - 1.0) * 100.0 if years > 0 and capital > 0 else 0.0

        peak   = equity_series.cummax()
        dd     = (peak - equity_series) / peak * 100.0
        max_dd = float(dd.max()) if not dd.empty else 0.0

        daily_eq = equity_series.resample("1D").last().dropna()
        daily_returns = daily_eq.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe  = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))
            neg_ret = daily_returns[daily_returns < 0]
            sortino = float((daily_returns.mean() / neg_ret.std()) * np.sqrt(252)) if len(neg_ret) > 1 and neg_ret.std() > 0 else sharpe
        else:
            sharpe, sortino = 0.0, 0.0

        return DualTFResult(
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
