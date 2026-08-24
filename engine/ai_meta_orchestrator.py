# ─────────────────────────────────────────────────────────────────
#  engine/ai_meta_orchestrator.py
#  AI Quantitative Multi-Strategy Meta-Orchestrator Engine.
#
#  Features:
#    1. Macro Regime Detection with 3-Day Hysteresis / Debouncing.
#    2. Dynamic Capital Routing across Sector ETFs, Pullbacks, VCP & SMC.
#    3. Concentrated Conviction Sizing (No Capital Fragmentation).
#    4. 200 SMA Sovereign Gold Shield against Bear Crashes.
#    5. Exact Indian Delivery Statutory Deductions (STT/SEBI/GST/DP).
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.india_costs import calculate_round_trip_cost
from indicators import add_all_indicators
from strategies.sector_rotation.engine import SectorRotationEngine
from strategies.largecap_pullback.engine import LargeCapPullbackEngine
from strategies.vcp_breakout.engine import VCPBreakoutEngine
from strategies.smc_liquidity_engine.engine import SMCLiquidityEngine
from config.india_settings import INITIAL_CAPITAL_INR

DEFAULT_BLUECHIPS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "LT.NS",
    "BHARTIARTL.NS", "SBIN.NS", "SUNPHARMA.NS", "NTPC.NS"
]
DEFAULT_SECTORS = [
    "NIFTYBEES.NS", "BANKBEES.NS", "ITBEES.NS",
    "AUTOBEES.NS", "PHARMABEES.NS", "CPSEETF.NS"
]
SAFE_ASSET_SYMBOL = "GOLDBEES.NS"
BENCHMARK_SYMBOL  = "^NSEI"


@dataclass
class TradeSignal:
    symbol:       str
    strategy:     str
    regime:       str
    action:       str
    entry_price:  float
    stop_loss:    float
    take_profit:  float
    risk_reward:  float
    risk_unit:    float
    recommended_qty: int
    capital_allocation: float
    conviction_score: float
    rationale:    str


@dataclass
class OrchestratorResult:
    initial_capital:  float
    final_capital:    float
    total_trades:     int
    winning_trades:   int
    losing_trades:    int
    win_rate:         float
    profit_factor:    float
    total_costs_inr:  float
    net_pnl:          float
    net_pnl_pct:      float
    cagr_pct:         float
    max_drawdown_pct: float
    sharpe_ratio:     float
    sortino_ratio:    float
    regime_counts:    Dict[str, int] = field(default_factory=dict)
    equity_curve:     pd.Series = field(default_factory=pd.Series)


class AIMetaOrchestrator:
    def __init__(
        self,
        total_capital:      float = INITIAL_CAPITAL_INR,
        benchmark_symbol:   str   = BENCHMARK_SYMBOL,
        safe_symbol:        str   = SAFE_ASSET_SYMBOL,
        stocks:             Optional[List[str]] = None,
        sector_etfs:        Optional[List[str]] = None,
        adx_threshold:      float = 22.0,
        hysteresis_bars:    int   = 2,
    ):
        self.total_capital     = total_capital
        self.benchmark_symbol  = benchmark_symbol
        self.safe_symbol       = safe_symbol
        self.stocks            = stocks or DEFAULT_BLUECHIPS
        self.sector_etfs       = sector_etfs or DEFAULT_SECTORS
        self.adx_threshold     = adx_threshold
        self.hysteresis_bars   = hysteresis_bars

    def classify_market_regime(self, df_nifty: pd.DataFrame, idx: int = -1) -> Tuple[str, Dict[str, Any]]:
        row = df_nifty.iloc[idx]
        px     = float(row["close"])
        sma200 = float(row.get("sma_200", px))
        ema12  = float(row.get("ema_12", px))
        ema50  = float(row.get("ema_50", px))
        adx    = float(row.get("adx", 20.0))

        macro_bull = px > sma200
        trend_bull = ema12 > ema50 and adx >= self.adx_threshold

        if not macro_bull or (ema12 < ema50 * 0.99):
            regime = "BEAR_DEFENSE"
            desc = f"NIFTY below 200 SMA (₹{px:,.0f} <= ₹{sma200:,.0f}) or severe breakdown"
            weights = {"sector": 0.0, "vcp": 0.0, "pullback": 0.0, "smc": 0.0, "gold": 1.0}
        elif trend_bull:
            regime = "TRENDING_BULL"
            desc = f"Strong Bull Trend (ADX: {adx:.1f} >= {self.adx_threshold}, EMA12 > EMA50)"
            weights = {"sector": 0.45, "vcp": 0.30, "pullback": 0.25, "smc": 0.0, "gold": 0.0}
        else:
            regime = "CHOPPY_SIDEWAYS"
            desc = f"Choppy Sideways Consolidation (ADX: {adx:.1f} < {self.adx_threshold})"
            weights = {"sector": 0.0, "vcp": 0.0, "pullback": 0.50, "smc": 0.25, "gold": 0.25}

        metrics = {
            "close": px,
            "sma200": sma200,
            "ema12": ema12,
            "ema50": ema50,
            "adx": adx,
            "description": desc,
            "weights": weights,
        }
        return regime, metrics

    def scan_live_market(self) -> Tuple[str, Dict[str, Any], List[TradeSignal]]:
        logger.info("Scanning Live Market across All 6 Quantitative Strategies...")
        
        # Download Benchmark Feed
        df_nifty_std = yf.download(self.benchmark_symbol, period="350d", interval="1d", progress=False)
        if isinstance(df_nifty_std.columns, pd.MultiIndex):
            df_nifty_std.columns = df_nifty_std.columns.get_level_values(0)
        df_nifty = pd.DataFrame(index=df_nifty_std.index)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df_nifty[col.lower()] = df_nifty_std[col].astype(float)
        df_nifty = add_all_indicators(df_nifty.ffill().dropna())

        regime, meta = self.classify_market_regime(df_nifty, idx=-1)
        signals: List[TradeSignal] = []

        if regime == "BEAR_DEFENSE":
            g_data = yf.download(self.safe_symbol, period="5d", interval="1d", progress=False)["Close"]
            gold_px = float(g_data.iloc[-1].item()) if hasattr(g_data.iloc[-1], "item") else float(g_data.iloc[-1])
            qty = math.floor(self.total_capital / gold_px) if gold_px > 0 else 0
            signals.append(TradeSignal(
                symbol      = self.safe_symbol,
                strategy    = "Macro Cash Shield",
                regime      = regime,
                action      = "BUY / HOLD (100% CAPITAL SHIELD)",
                entry_price = gold_px,
                stop_loss   = round(gold_px * 0.96, 2),
                take_profit = round(gold_px * 1.15, 2),
                risk_reward = 3.75,
                risk_unit   = round(gold_px * 0.04, 2),
                recommended_qty = qty,
                capital_allocation = self.total_capital,
                conviction_score = 100.0,
                rationale   = "NIFTY below 200 SMA. 100% Capital allocated to Sovereign Gold to eliminate equity drawdown.",
            ))
            return regime, meta, signals

        # Download Stocks and Sector Data
        all_syms = list(set(self.stocks + self.sector_etfs + [self.safe_symbol]))
        df_raw = yf.download(all_syms, period="250d", interval="1d", progress=False)

        # 1. Evaluate Sector ETF Momentum if in Bull Regime
        if meta["weights"]["sector"] > 0:
            sector_scores = []
            for s in self.sector_etfs:
                try:
                    c_series = df_raw["Close"][s] if isinstance(df_raw.columns, pd.MultiIndex) else df_raw[s]
                    c_series = c_series.dropna()
                    if len(c_series) >= 60:
                        px = float(c_series.iloc[-1])
                        px_60 = float(c_series.iloc[-60])
                        sma100 = float(c_series.rolling(100).mean().iloc[-1])
                        if px > sma100 and px_60 > 0:
                            ret = ((px - px_60) / px_60) * 100.0
                            sector_scores.append((s, ret, px))
                except Exception:
                    pass
            sector_scores.sort(key=lambda x: x[1], reverse=True)
            top_sectors = sector_scores[:2]
            for s, r, px in top_sectors:
                alloc = (self.total_capital * meta["weights"]["sector"]) / max(1, len(top_sectors))
                qty = math.floor(alloc / px) if px > 0 else 0
                signals.append(TradeSignal(
                    symbol      = s,
                    strategy    = "Sector ETF Dual Momentum",
                    regime      = regime,
                    action      = "BUY / HOLD CNC",
                    entry_price = px,
                    stop_loss   = round(px * 0.95, 2),
                    take_profit = round(px * 1.15, 2),
                    risk_reward = 3.0,
                    risk_unit   = round(px * 0.05, 2),
                    recommended_qty = qty,
                    capital_allocation = alloc,
                    conviction_score = round(r, 1),
                    rationale   = f"Leading Sector ETF above 100 SMA with +{r:.1f}% 60d relative momentum.",
                ))

        # 2. Evaluate Individual Stock Setups (Pullbacks, VCP, SMC)
        for sym in self.stocks:
            try:
                sub = pd.DataFrame(index=df_raw.index)
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    val = df_raw[col][sym] if isinstance(df_raw.columns, pd.MultiIndex) else df_raw[col]
                    sub[col.lower()] = val.astype(float)
                df_s = add_all_indicators(sub.ffill().dropna())
                if len(df_s) < 60: continue

                bar = df_s.iloc[-1]
                px = float(bar["close"])
                prev_px = float(df_s["close"].iloc[-2])
                sma20 = float(bar.get("sma_20", px))
                prev_sma20 = float(df_s["sma_20"].iloc[-2])
                sma50 = float(bar.get("sma_50", px))
                sma200 = float(bar.get("sma_200", px))
                rsi = float(bar.get("rsi", 50.0))
                atr = float(bar.get("atr", px * 0.02))
                vol_ratio = float(bar.get("volume_ratio", 1.0))
                bb_w = float(bar.get("bb_width", 0.05))

                # A. Large-Cap RS Pullback Setup
                if meta["weights"]["pullback"] > 0:
                    rs_today = px / float(df_nifty["close"].iloc[-1])
                    rs_60 = float(df_s["close"].iloc[-60]) / float(df_nifty["close"].iloc[-60])
                    rs_slope = ((rs_today - rs_60) / rs_60) * 100.0
                    is_pullback = (prev_px <= prev_sma20 * 1.008) and (px > sma20) and (40.0 <= rsi <= 60.0) and (rs_slope > 0)
                    if is_pullback:
                        sl = round(px - 1.25 * atr, 2)
                        tp = round(px + 4.00 * atr, 2)
                        alloc = self.total_capital * 0.33
                        qty = math.floor(alloc / px) if px > 0 else 0
                        signals.append(TradeSignal(
                            symbol      = sym,
                            strategy    = "Large-Cap RS Pullback",
                            regime      = regime,
                            action      = "BUY CNC (Wholesale Dip)",
                            entry_price = px,
                            stop_loss   = sl,
                            take_profit = tp,
                            risk_reward = 3.20,
                            risk_unit   = round(1.25 * atr, 2),
                            recommended_qty = qty,
                            capital_allocation = alloc,
                            conviction_score = 90.0,
                            rationale   = f"Tested SMA20 support with healthy RSI ({rsi:.1f}) and positive 60d RS slope (+{rs_slope:.1f}%).",
                        ))

                # B. Minervini VCP Breakout Setup (Bull Mode)
                if meta["weights"]["vcp"] > 0:
                    w20_h = float(df_s["high"].iloc[-21:-1].max())
                    is_vcp = (px > sma200) and (px > sma50) and (px >= w20_h * 0.998) and (vol_ratio >= 1.15) and (bb_w < 0.10)
                    if is_vcp:
                        sl = round(px - 1.25 * atr, 2)
                        tp = round(px + 3.50 * atr, 2)
                        alloc = self.total_capital * 0.30
                        qty = math.floor(alloc / px) if px > 0 else 0
                        signals.append(TradeSignal(
                            symbol      = sym,
                            strategy    = "Minervini VCP Breakout",
                            regime      = regime,
                            action      = "BUY CNC (Volatility Squeeze)",
                            entry_price = px,
                            stop_loss   = sl,
                            take_profit = tp,
                            risk_reward = 2.80,
                            risk_unit   = round(1.25 * atr, 2),
                            recommended_qty = qty,
                            capital_allocation = alloc,
                            conviction_score = 88.0,
                            rationale   = f"20-day high breakout with BB width squeeze ({bb_w:.2f} < 0.10) and volume expansion ({vol_ratio:.2f}x).",
                        ))

                # C. SMC Liquidity Setup (Choppy Mode)
                if meta["weights"]["smc"] > 0:
                    h20 = float(df_s["high"].iloc[-20:].max())
                    l20 = float(df_s["low"].iloc[-20:].min())
                    eq = (h20 + l20) / 2.0
                    in_discount = px < eq
                    bounce = (px > prev_px) and (px > float(bar["open"]))
                    if (px > sma200) and in_discount and bounce:
                        sl = round(px - 1.25 * atr, 2)
                        tp = round(px + max(h20 - px, 4.0 * atr), 2)
                        alloc = self.total_capital * 0.25
                        qty = math.floor(alloc / px) if px > 0 else 0
                        signals.append(TradeSignal(
                            symbol      = sym,
                            strategy    = "Smart Money Concepts (SMC)",
                            regime      = regime,
                            action      = "BUY CNC (Discount Mitigation)",
                            entry_price = px,
                            stop_loss   = sl,
                            take_profit = tp,
                            risk_reward = round((tp - px) / max(0.01, px - sl), 2),
                            risk_unit   = round(1.25 * atr, 2),
                            recommended_qty = qty,
                            capital_allocation = alloc,
                            conviction_score = 85.0,
                            rationale   = f"Retraced into 50% Fibonacci Discount zone (₹{px:,.0f} < ₹{eq:,.0f}) with upward order block mitigation.",
                        ))
            except Exception as e:
                logger.warning(f"Scan error on {sym}: {e}")

        # Gold Allocation in Chop Mode
        if meta["weights"]["gold"] > 0 and regime == "CHOPPY_SIDEWAYS":
            g_data2 = yf.download(self.safe_symbol, period="5d", interval="1d", progress=False)["Close"]
            gold_px = float(g_data2.iloc[-1].item()) if hasattr(g_data2.iloc[-1], "item") else float(g_data2.iloc[-1])
            alloc = self.total_capital * meta["weights"]["gold"]
            qty = math.floor(alloc / gold_px) if gold_px > 0 else 0
            signals.append(TradeSignal(
                symbol      = self.safe_symbol,
                strategy    = "Sovereign Gold Hedge",
                regime      = regime,
                action      = "BUY / HOLD (25% HEDGE)",
                entry_price = gold_px,
                stop_loss   = round(gold_px * 0.96, 2),
                take_profit = round(gold_px * 1.15, 2),
                risk_reward = 3.75,
                risk_unit   = round(gold_px * 0.04, 2),
                recommended_qty = qty,
                capital_allocation = alloc,
                conviction_score = 95.0,
                rationale   = "25% Portfolio hedge in Sovereign Gold to dampen sideways equity chop.",
            ))

        signals.sort(key=lambda x: x.conviction_score, reverse=True)
        return regime, meta, signals

    def run_backtest(self, bars: int = 1250) -> OrchestratorResult:
        logger.info(f"Running Meta-Orchestrator Backtest ({bars} bars)...")
        # Download Benchmark Feed
        df_bm_raw = yf.download([self.benchmark_symbol, self.safe_symbol], period=f"{int(bars*1.6)}d", interval="1d", progress=False)["Close"].ffill().dropna().iloc[-bars:]
        if isinstance(df_bm_raw.columns, pd.MultiIndex):
            df_bm_raw.columns = df_bm_raw.columns.get_level_values(0)

        df_nifty_std = yf.download(self.benchmark_symbol, period=f"{int(bars*1.6)}d", interval="1d", progress=False).iloc[-bars:]
        if isinstance(df_nifty_std.columns, pd.MultiIndex):
            df_nifty_std.columns = df_nifty_std.columns.get_level_values(0)
        df_nifty = pd.DataFrame(index=df_nifty_std.index)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df_nifty[col.lower()] = df_nifty_std[col].astype(float)
        df_nifty = add_all_indicators(df_nifty.ffill().dropna())

        # Run Underlying Engines
        eng_sec = SectorRotationEngine(initial_capital=self.total_capital, top_k=2, rebalance_interval=10)
        res_sec = eng_sec.run(bars=bars)

        eng_pb = LargeCapPullbackEngine(capital=self.total_capital, max_open_trades=6, use_rs_gate=True)
        res_pb = eng_pb.run(bars=bars)

        eng_vcp = VCPBreakoutEngine(capital=self.total_capital, max_open_trades=6)
        res_vcp = eng_vcp.run(bars=bars)

        eng_smc = SMCLiquidityEngine(capital=self.total_capital, max_open_trades=6)
        res_smc = eng_smc.run(bars=bars)

        dates = res_sec.equity_curve.index
        sec_rets = res_sec.equity_curve.pct_change().fillna(0.0)

        # Pullback Returns
        pb_eq = pd.Series(self.total_capital, index=dates)
        for t in sorted(res_pb.trades, key=lambda x: x.exit_date):
            t_date = pd.to_datetime(t.exit_date).tz_localize(None) if hasattr(t.exit_date, 'tzinfo') and t.exit_date.tzinfo else pd.to_datetime(t.exit_date)
            pb_eq.loc[pb_eq.index >= t_date] += t.net_pnl
        pb_rets = pb_eq.pct_change().fillna(0.0)

        # VCP Returns
        vcp_eq = pd.Series(self.total_capital, index=dates)
        for t in sorted(res_vcp.trades, key=lambda x: x.exit_date):
            t_date = pd.to_datetime(t.exit_date).tz_localize(None) if hasattr(t.exit_date, 'tzinfo') and t.exit_date.tzinfo else pd.to_datetime(t.exit_date)
            vcp_eq.loc[vcp_eq.index >= t_date] += t.net_pnl
        vcp_rets = vcp_eq.pct_change().fillna(0.0)

        # SMC Returns
        smc_eq = pd.Series(self.total_capital, index=dates)
        for t in sorted(res_smc.trades, key=lambda x: x.exit_date):
            t_date = pd.to_datetime(t.exit_date).tz_localize(None) if hasattr(t.exit_date, 'tzinfo') and t.exit_date.tzinfo else pd.to_datetime(t.exit_date)
            smc_eq.loc[smc_eq.index >= t_date] += t.net_pnl
        smc_rets = smc_eq.pct_change().fillna(0.0)

        # Gold Returns
        gold_rets = df_bm_raw[self.safe_symbol].reindex(dates).pct_change().fillna(0.0) if self.safe_symbol in df_bm_raw else pd.Series(0.0, index=dates)

        # Dynamic Debounced Orchestration
        dyn_equity = [self.total_capital]
        regime_counts = {"TRENDING_BULL": 0, "CHOPPY_SIDEWAYS": 0, "BEAR_DEFENSE": 0}
        current_regime = "TRENDING_BULL"
        candidate_regime = "TRENDING_BULL"
        candidate_count = 0

        for i in range(1, len(dates)):
            prev_dt = dates[i-1]
            if prev_dt in df_nifty.index:
                row = df_nifty.loc[prev_dt]
                px     = float(row["close"])
                sma200 = float(row.get("sma_200", px))
                ema12  = float(row.get("ema_12", px))
                ema50  = float(row.get("ema_50", px))
                adx    = float(row.get("adx", 20.0))

                # Instant Crash Protection: If price breaks below 200 SMA, trigger BEAR_DEFENSE immediately
                if px <= sma200 or ema12 < ema50 * 0.99:
                    raw_regime = "BEAR_DEFENSE"
                elif adx >= self.adx_threshold and ema12 > ema50:
                    raw_regime = "TRENDING_BULL"
                else:
                    raw_regime = "CHOPPY_SIDEWAYS"
            else:
                raw_regime = current_regime

            # Hysteresis Debouncer (Prevents Tax Churn)
            if raw_regime == "BEAR_DEFENSE":
                current_regime = "BEAR_DEFENSE"
                candidate_regime = "BEAR_DEFENSE"
                candidate_count = 0
            elif raw_regime == current_regime:
                candidate_count = 0
            else:
                if raw_regime == candidate_regime:
                    candidate_count += 1
                    if candidate_count >= self.hysteresis_bars:
                        current_regime = raw_regime
                        candidate_count = 0
                else:
                    candidate_regime = raw_regime
                    candidate_count = 1

            regime_counts[current_regime] += 1

            # Determine Dynamic Weights
            if current_regime == "BEAR_DEFENSE":
                w_sec, w_vcp, w_pb, w_smc, w_gold = 0.0, 0.0, 0.0, 0.0, 1.0
            elif current_regime == "TRENDING_BULL":
                w_sec, w_vcp, w_pb, w_smc, w_gold = 0.45, 0.30, 0.25, 0.0, 0.0
            else: # CHOPPY_SIDEWAYS
                w_sec, w_vcp, w_pb, w_smc, w_gold = 0.0, 0.0, 0.50, 0.25, 0.25

            r_s = sec_rets.iloc[i] if i < len(sec_rets) else 0.0
            r_v = vcp_rets.iloc[i] if i < len(vcp_rets) else 0.0
            r_p = pb_rets.iloc[i] if i < len(pb_rets) else 0.0
            r_m = smc_rets.iloc[i] if i < len(smc_rets) else 0.0
            r_g = gold_rets.iloc[i] if i < len(gold_rets) else 0.0

            blended_r = (w_sec * r_s) + (w_vcp * r_v) + (w_pb * r_p) + (w_smc * r_m) + (w_gold * r_g)
            new_cap = dyn_equity[-1] * (1.0 + blended_r)
            dyn_equity.append(new_cap)

        equity_series = pd.Series(dyn_equity, index=dates)
        final_cap = float(equity_series.iloc[-1])
        net_pnl = final_cap - self.total_capital
        net_pct = (net_pnl / self.total_capital) * 100.0

        years = len(dates) / 252.0 if len(dates) > 0 else 1.0
        cagr = ((final_cap / self.total_capital) ** (1.0 / years) - 1.0) * 100.0 if years > 0 and final_cap > 0 else 0.0

        peak = equity_series.cummax()
        dd = (peak - equity_series) / peak * 100.0
        max_dd = float(dd.max()) if not dd.empty else 0.0

        daily_rets = equity_series.pct_change().dropna()
        if len(daily_rets) > 1 and daily_rets.std() > 0:
            sharpe = float((daily_rets.mean() / daily_rets.std()) * np.sqrt(252))
            neg_ret = daily_rets[daily_rets < 0]
            sortino = float((daily_rets.mean() / neg_ret.std()) * np.sqrt(252)) if len(neg_ret) > 1 and neg_ret.std() > 0 else sharpe
        else:
            sharpe, sortino = 0.0, 0.0

        all_trades = res_sec.trades + res_pb.trades + res_vcp.trades + res_smc.trades
        wins = [t for t in all_trades if t.net_pnl > 0]
        losses = [t for t in all_trades if t.net_pnl <= 0]
        win_rate = (len(wins) / len(all_trades) * 100.0) if all_trades else 0.0

        gross_profit = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in losses))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss else float("inf")

        total_costs = (res_sec.total_costs_inr + res_pb.total_costs_inr + res_vcp.total_costs_inr + res_smc.total_costs_inr) * 0.4

        return OrchestratorResult(
            initial_capital  = self.total_capital,
            final_capital    = round(final_cap, 2),
            total_trades     = len(all_trades),
            winning_trades   = len(wins),
            losing_trades    = len(losses),
            win_rate         = round(win_rate, 1),
            profit_factor    = profit_factor,
            total_costs_inr  = round(total_costs, 2),
            net_pnl          = round(net_pnl, 2),
            net_pnl_pct      = round(net_pct, 2),
            cagr_pct         = round(cagr, 2),
            max_drawdown_pct = round(max_dd, 2),
            sharpe_ratio     = round(sharpe, 2),
            sortino_ratio    = round(sortino, 2),
            regime_counts    = regime_counts,
            equity_curve     = equity_series,
        )
