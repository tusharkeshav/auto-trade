# ─────────────────────────────────────────────────────────────────
#  probability/hmm_multiplexer.py
#  Adaptive Strategy Selection Engine (HMM Strategy Multiplexer).
#
#  Mathematical & Architectural Basis:
#    - Uses GaussianHMM posterior state probabilities (P_0, P_1, P_2)
#      to dynamically allocate capital across specialized strategies.
#
#    - Regime 0 (Calm / Bull Accumulation - P_0):
#      Dispatches to Mean Reversion (ConnorsScorer / RSI-2 pullback).
#
#    - Regime 1 (Trending / Range - P_1):
#      Dispatches to Trend-Pullback Momentum (UnifiedCrossScorer Type A).
#
#    - Regime 2 (Crash / Extreme Volatility - P_2):
#      Modulates risk sizing scalar: Scalar = max(0.40, 1.0 - P_2).
#      At 100% crash probability, capital allocation scales down to 0.8% risk.
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional

import pandas as pd
from loguru import logger

from probability.signal_scorer       import TradeSignal
from probability.connors_scorer      import ConnorsScorer
from probability.unified_cross_scorer import UnifiedCrossScorer
from config.india_settings           import INDIA_SIGNAL_THRESHOLD

IST = ZoneInfo("Asia/Kolkata")


class HMMStrategyMultiplexer:
    """
    Adaptive Strategy Selection Engine for Indian equity indices.
    """

    def __init__(
        self,
        symbol:       str   = "NIFTY50",
        threshold:    float = INDIA_SIGNAL_THRESHOLD,
        current_vix:  float = 15.0,
        interval:     str   = "1d",
        atr_sl_mult:  float = 1.0,
        atr_tp_mult:  float = 3.0,
    ):
        self.symbol      = symbol
        self.threshold   = threshold
        self.current_vix = current_vix
        self.interval    = interval
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult

        # Sub-Scorers specialized by regime
        self.mr_scorer    = ConnorsScorer(
            symbol       = symbol,
            threshold    = threshold,
            interval     = interval,
        )
        self.trend_scorer = UnifiedCrossScorer(
            symbol       = symbol,
            threshold    = threshold,
            atr_sl_mult  = atr_sl_mult,
            atr_tp_mult  = atr_tp_mult,
            interval     = interval,
            current_vix  = current_vix,
        )

    def set_vix(self, vix: float) -> None:
        """Update VIX dynamically during backtest or live loop."""
        self.current_vix = vix
        if hasattr(self.trend_scorer, "current_vix"):
            self.trend_scorer.current_vix = vix

    def score(self, row: pd.Series, df_slice: Optional[pd.DataFrame] = None) -> TradeSignal:
        """
        Score candle by evaluating HMM posterior probabilities and multiplexing sub-scorers.
        """
        price = float(row["close"])
        if hasattr(row, "get") and row.get("vix") is not None and not math.isnan(float(row.get("vix"))):
            self.set_vix(float(row.get("vix")))

        # ── 1. Extract HMM Posterior Regime Probabilities ───────────
        p0 = float(row.get("hmm_prob_0", 1.0))  # Regime 0: Calm / Bull Accumulation (Mean Reversion zone)
        p1 = float(row.get("hmm_prob_1", 0.0))  # Regime 1: Trending / Range (Trend-Pullback zone)
        p2 = float(row.get("hmm_prob_2", 0.0))  # Regime 2: Crash / Extreme Volatility (Risk Dampener)

        # ── 2. Evaluate Specialized Sub-Scorers ─────────────────────
        sig_mr    = self.mr_scorer.score(row, df_slice)
        sig_trend = self.trend_scorer.score(row, df_slice)

        mr_active    = sig_mr.is_tradeable() and sig_mr.direction == "LONG"
        trend_active = sig_trend.is_tradeable() and sig_trend.direction == "LONG"

        if not mr_active and not trend_active:
            return self._no_trade(
                row, price,
                f"[HMM MULTIPLEXER | P0={p0:.0%}, P1={p1:.0%}, P2={p2:.0%}] Neither MR nor Trend triggered"
            )

        # ── 3. Probabilistic Strategy Allocation & Blending ─────────
        # Weight each strategy's conviction by our confidence in its target regime
        score_mr    = (p0 * sig_mr.probability) if mr_active else 0.0
        score_trend = (p1 * sig_trend.probability) if trend_active else 0.0

        if mr_active and trend_active:
            chosen_sig = sig_mr if score_mr >= score_trend else sig_trend
            source_lbl = "MR (Connors RSI-2)" if chosen_sig is sig_mr else "TREND (UnifiedCross)"
            blended_prob = round(((p0 * sig_mr.probability) + (p1 * sig_trend.probability)) / max(0.01, p0 + p1), 1)
        elif mr_active:
            chosen_sig = sig_mr
            source_lbl = "MR (Connors RSI-2)"
            blended_prob = round(sig_mr.probability * (1.0 - (0.5 * p2)), 1)
        else:
            chosen_sig = sig_trend
            source_lbl = "TREND (UnifiedCross)"
            blended_prob = round(sig_trend.probability * (1.0 - (0.5 * p2)), 1)

        # Final check against institutional threshold
        if blended_prob < self.threshold:
            return self._no_trade(
                row, price,
                f"[HMM MULTIPLEXER | {source_lbl}] Blended prob {blended_prob}% < threshold {self.threshold}%"
            )

        sizing_scalar = round(max(0.40, 1.0 - p2), 2)
        return replace(
            chosen_sig,
            probability = blended_prob,
            risk_amount = sizing_scalar,
            confidence  = "HIGH" if blended_prob >= 70 else "MEDIUM",
            reason      = f"[HMM ADAPTIVE | P0={p0:.0%}, P1={p1:.0%}, P2={p2:.0%} | Scalar={sizing_scalar:.2f}] Selected {source_lbl} (Prob: {blended_prob}%) | {chosen_sig.reason}"
        )

    def _no_trade(self, row: pd.Series, price: float, reason: str) -> TradeSignal:
        return TradeSignal(
            symbol       = self.symbol,
            timestamp    = row.name if hasattr(row, "name") and row.name is not None else datetime.now(IST),
            direction    = "NO_TRADE",
            probability  = 50.0,
            confidence   = "LOW",
            raw_score    = 0.0,
            entry_price  = round(price, 2),
            stop_loss    = round(price, 2),
            take_profit1 = round(price, 2),
            take_profit2 = round(price, 2),
            risk_amount  = 0.0,
            reason       = reason,
        )
