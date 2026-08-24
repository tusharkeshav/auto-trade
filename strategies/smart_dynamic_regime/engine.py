# ─────────────────────────────────────────────────────────────────
#  strategies/smart_dynamic_regime/engine.py
#  Dynamic Macro-Regime Capital Allocator Engine.
#
#  Dynamically shifts capital between:
#    1. TRENDING_BULL : 60% Sector ETF Momentum + 40% Large-Cap Stocks
#    2. CHOPPY_RANGE  : 70% Large-Cap Pullbacks (60d RS Gate) + 30% Gold
#    3. BEAR_DEFENSE  : 100% Sovereign Gold (GOLDBEES) / Liquid Cash
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from strategies.sector_rotation.engine import SectorRotationEngine
from strategies.largecap_pullback.engine import LargeCapPullbackEngine
from indicators import add_all_indicators
from config.india_settings import INITIAL_CAPITAL_INR

SAFE_ASSET_SYMBOL = "GOLDBEES.NS"
BENCHMARK_SYMBOL  = "^NSEI"


@dataclass
class DynamicRegimeResult:
    initial_capital: float
    final_capital:   float
    total_trades:    int
    net_pnl:         float
    net_pnl_pct:     float
    cagr_pct:        float
    max_drawdown_pct:float
    sharpe_ratio:    float
    sortino_ratio:   float
    total_costs_inr: float
    regime_counts:   Dict[str, int] = field(default_factory=dict)
    equity_curve:    pd.Series = field(default_factory=pd.Series)


class DynamicRegimeAllocatorEngine:
    def __init__(
        self,
        total_capital: float = INITIAL_CAPITAL_INR,
        benchmark_symbol: str = BENCHMARK_SYMBOL,
        safe_symbol: str = SAFE_ASSET_SYMBOL,
        adx_threshold: float = 22.0,
    ):
        self.total_capital = total_capital
        self.benchmark_symbol = benchmark_symbol
        self.safe_symbol = safe_symbol
        self.adx_threshold = adx_threshold

    def run(self, bars: int = 1250) -> DynamicRegimeResult:
        # 1. Fetch Benchmark & Gold Data
        df_bm_raw = yf.download([self.benchmark_symbol, self.safe_symbol], period=f"{int(bars*1.6)}d", interval="1d", progress=False)["Close"].ffill().dropna().iloc[-bars:]
        if isinstance(df_bm_raw.columns, pd.MultiIndex):
            df_bm_raw.columns = df_bm_raw.columns.get_level_values(0)

        # Prepare Indicators for Benchmark
        df_nifty_std = yf.download(self.benchmark_symbol, period=f"{int(bars*1.6)}d", interval="1d", progress=False).iloc[-bars:]
        if isinstance(df_nifty_std.columns, pd.MultiIndex):
            df_nifty_std.columns = df_nifty_std.columns.get_level_values(0)
        df_nifty = pd.DataFrame(index=df_nifty_std.index)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df_nifty[col.lower()] = df_nifty_std[col].astype(float)
        df_nifty = add_all_indicators(df_nifty.ffill().dropna())

        # 2. Run Underlying Strategy Engines on Full Capital Bases
        eng_sector = SectorRotationEngine(initial_capital=self.total_capital, top_k=2, rebalance_interval=10)
        res_sector = eng_sector.run(bars=bars)

        eng_pullback = LargeCapPullbackEngine(capital=self.total_capital, max_open_trades=6, use_rs_gate=True)
        res_pullback = eng_pullback.run(bars=bars)

        # 3. Synchronize Timelines
        dates = res_sector.equity_curve.index
        df_sim = pd.DataFrame(index=dates)

        # Returns series
        sec_rets = res_sector.equity_curve.pct_change().fillna(0.0)

        pb_eq = pd.Series(self.total_capital, index=dates)
        for t in sorted(res_pullback.trades, key=lambda x: x.exit_date):
            t_date = pd.to_datetime(t.exit_date).tz_localize(None) if hasattr(t.exit_date, 'tzinfo') and t.exit_date.tzinfo else pd.to_datetime(t.exit_date)
            pb_eq.loc[pb_eq.index >= t_date] += t.net_pnl
        pb_rets = pb_eq.pct_change().fillna(0.0)

        gold_rets = df_bm_raw[self.safe_symbol].reindex(dates).pct_change().fillna(0.0) if self.safe_symbol in df_bm_raw else pd.Series(0.0, index=dates)

        # 4. Simulate Dynamic Capital Shift
        regime_counts = {"TRENDING_BULL": 0, "CHOPPY_RANGE": 0, "BEAR_DEFENSE": 0}
        dyn_equity = [self.total_capital]

        for i in range(1, len(dates)):
            dt = dates[i]
            prev_dt = dates[i-1]

            # Evaluate Regime on Day i-1 Close
            nifty_row = df_nifty.loc[prev_dt] if prev_dt in df_nifty.index else df_nifty.iloc[-1]
            px = float(nifty_row["close"])
            sma200 = float(nifty_row.get("sma_200", px))
            ema12 = float(nifty_row.get("ema_12", px))
            ema50 = float(nifty_row.get("ema_50", px))
            adx = float(nifty_row.get("adx", 20.0))

            if px <= sma200 or ema12 < ema50 * 0.99:
                regime = "BEAR_DEFENSE"
                w_sec, w_pb, w_gold = 0.0, 0.0, 1.0
            elif adx >= self.adx_threshold and ema12 > ema50:
                regime = "TRENDING_BULL"
                w_sec, w_pb, w_gold = 0.60, 0.40, 0.0
            else:
                regime = "CHOPPY_RANGE"
                w_sec, w_pb, w_gold = 0.0, 0.70, 0.30

            regime_counts[regime] += 1

            # Day i Blended Daily Return
            r_sec = sec_rets.iloc[i] if i < len(sec_rets) else 0.0
            r_pb  = pb_rets.iloc[i] if i < len(pb_rets) else 0.0
            r_gold = gold_rets.iloc[i] if i < len(gold_rets) else 0.0

            blended_r = (w_sec * r_sec) + (w_pb * r_pb) + (w_gold * r_gold)
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

        total_costs = (res_sector.total_costs_inr + res_pullback.total_costs_inr) * 0.5
        total_trades = res_sector.total_trades + res_pullback.total_trades

        return DynamicRegimeResult(
            initial_capital  = self.total_capital,
            final_capital    = round(final_cap, 2),
            total_trades     = total_trades,
            net_pnl          = round(net_pnl, 2),
            net_pnl_pct      = round(net_pct, 2),
            cagr_pct         = round(cagr, 2),
            max_drawdown_pct = round(max_dd, 2),
            sharpe_ratio     = round(sharpe, 2),
            sortino_ratio    = round(sortino, 2),
            total_costs_inr  = round(total_costs, 2),
            regime_counts    = regime_counts,
            equity_curve     = equity_series,
        )
