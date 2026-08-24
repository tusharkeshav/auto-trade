# ─────────────────────────────────────────────────────────────────
#  strategies/all_weather_dual_book/engine.py
#  Synchronized 50/50 Dual-Book Engine (Sector ETF + Stock Pullbacks).
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from strategies.sector_rotation.engine import SectorRotationEngine
from strategies.largecap_pullback.engine import LargeCapPullbackEngine
from config.india_settings import INITIAL_CAPITAL_INR


@dataclass
class DualBookResult:
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
    equity_curve:    pd.Series = field(default_factory=pd.Series)


class AllWeatherDualBookEngine:
    def __init__(
        self,
        total_capital: float = INITIAL_CAPITAL_INR,
        book1_weight:  float = 0.50,
        book2_weight:  float = 0.50,
    ):
        self.total_capital = total_capital
        self.book1_capital = total_capital * book1_weight
        self.book2_capital = total_capital * book2_weight

    def run(self, bars: int = 1250) -> DualBookResult:
        eng_sector = SectorRotationEngine(initial_capital=self.book1_capital, top_k=2, rebalance_interval=10)
        res_sector = eng_sector.run(bars=bars)

        eng_pullback = LargeCapPullbackEngine(capital=self.book2_capital, max_open_trades=6, use_rs_gate=True)
        res_pullback = eng_pullback.run(bars=bars)

        # Synchronize daily equity curves
        common_dates = res_sector.equity_curve.index
        df_comb = pd.DataFrame(index=common_dates)
        df_comb["sector_eq"] = res_sector.equity_curve

        stock_eq = pd.Series(self.book2_capital, index=common_dates)
        for t in sorted(res_pullback.trades, key=lambda x: x.exit_date):
            t_date = pd.to_datetime(t.exit_date).tz_localize(None) if hasattr(t.exit_date, 'tzinfo') and t.exit_date.tzinfo else pd.to_datetime(t.exit_date)
            stock_eq.loc[stock_eq.index >= t_date] += t.net_pnl
        df_comb["stock_eq"] = stock_eq
        df_comb["comb_eq"]  = df_comb["sector_eq"] + df_comb["stock_eq"]

        final_cap = float(df_comb["comb_eq"].iloc[-1])
        net_pnl   = final_cap - self.total_capital
        net_pct   = (net_pnl / self.total_capital) * 100.0

        years = len(df_comb) / 252.0 if len(df_comb) > 0 else 1.0
        cagr  = ((final_cap / self.total_capital) ** (1.0 / years) - 1.0) * 100.0 if years > 0 and final_cap > 0 else 0.0

        peak   = df_comb["comb_eq"].cummax()
        dd     = (peak - df_comb["comb_eq"]) / peak * 100.0
        max_dd = float(dd.max()) if not dd.empty else 0.0

        daily_rets = df_comb["comb_eq"].pct_change().dropna()
        if len(daily_rets) > 1 and daily_rets.std() > 0:
            sharpe  = float((daily_rets.mean() / daily_rets.std()) * np.sqrt(252))
            neg_ret = daily_rets[daily_rets < 0]
            sortino = float((daily_rets.mean() / neg_ret.std()) * np.sqrt(252)) if len(neg_ret) > 1 and neg_ret.std() > 0 else sharpe
        else:
            sharpe, sortino = 0.0, 0.0

        total_costs = res_sector.total_costs_inr + res_pullback.total_costs_inr
        total_trades = res_sector.total_trades + res_pullback.total_trades

        return DualBookResult(
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
            equity_curve     = df_comb["comb_eq"],
        )
