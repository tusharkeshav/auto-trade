# ─────────────────────────────────────────────────────────────────
#  probability/regime_switch_scorer.py
#  VIX regime-switching equity strategy scorer for NIFTY/BANKNIFTY.
#
#  Dynamically switches or blends between Trend-Following (Momentum)
#  and Mean Reversion based on India VIX volatility zones:
#
#    - VIX < 18.0 (Low Vol):
#      Momentum regime. Markets trend cleanly with institutional flow.
#      Dispatches to IndiaMomentumScorer.
#
#    - VIX 18.0 to 25.0 (Elevated/Transition):
#      Hybrid regime. Evaluates both Momentum and Mean Reversion.
#      If both agree on direction, takes weighted average probability.
#      If only one fires or they disagree, takes stronger signal with a
#      0.85× probability dampener (reflecting transition uncertainty).
#
#    - VIX > 25.0 (High Vol / Stress):
#      Mean Reversion regime. Extreme moves overshoot and snap back.
#      Dispatches to IndiaSignalScorer (mean reversion).
#
#  Research basis:
#    - Daniel & Moskowitz (2016): Momentum crashes occur when VIX is high.
#    - Barroso & Santa-Clara (2015): Restricting momentum to low VIX
#      doubles Sharpe ratio and cuts max drawdown from 76% to 29%.
#    - Whaley (2000, 2009): VIX > 20-25 signals oversold over-reaction,
#      creating high-probability contrarian / mean-reversion setups.
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

from probability.signal_scorer        import TradeSignal
from probability.india_signal_scorer  import IndiaSignalScorer
from probability.india_momentum_scorer import IndiaMomentumScorer
from config.india_settings            import (
    INDIA_SIGNAL_THRESHOLD,
    VIX_NORMAL_HIGH,
    VIX_STRESS,
)

IST = ZoneInfo("Asia/Kolkata")


class RegimeSwitchScorer:
    """
    VIX regime-switching dispatcher for Indian equity indices.
    """

    def __init__(
        self,
        symbol:       str   = "NIFTY50",
        threshold:    float = INDIA_SIGNAL_THRESHOLD,
        current_vix:  float = 15.0,
        interval:     str   = "15m",
    ):
        self.symbol      = symbol
        self.threshold   = threshold
        self.current_vix = current_vix
        self.interval    = interval

        # Sub-scorers
        self.mr_scorer  = IndiaSignalScorer(
            symbol          = symbol,
            long_threshold  = threshold,
            short_threshold = 100 - threshold,
            current_vix     = current_vix,
            interval        = interval,
        )
        self.mom_scorer = IndiaMomentumScorer(
            symbol          = symbol,
            threshold       = threshold,
            interval        = interval,
        )

    def set_vix(self, vix: float) -> None:
        """Update VIX dynamically during backtest or live loop."""
        self.current_vix = vix
        self.mr_scorer.current_vix = vix

    def score(self, row: pd.Series, df_slice: pd.DataFrame | None = None) -> TradeSignal:
        """
        Score a candle by evaluating VIX regime and dispatching to appropriate scorer.

        If row contains a 'vix' column, dynamically updates current_vix.
        """
        price = float(row["close"])
        if hasattr(row, "get") and row.get("vix") is not None and not math.isnan(float(row.get("vix"))):
            self.set_vix(float(row.get("vix")))

        vix = self.current_vix

        # ── 1. Low VIX Regime (< 18.0): Pure Momentum ─────────────────
        if vix < VIX_NORMAL_HIGH:
            sig = self.mom_scorer.score(row, df_slice)
            if sig.is_tradeable():
                return replace(
                    sig,
                    reason=f"[MOMENTUM REGIME | VIX={vix:.1f} < {VIX_NORMAL_HIGH}] {sig.reason}"
                )
            return sig

        # ── 2. High VIX Regime (> 25.0): Pure Mean Reversion ──────────
        if vix >= VIX_STRESS:
            sig = self.mr_scorer.score(row)
            if sig.is_tradeable():
                return replace(
                    sig,
                    reason=f"[MEAN_REV REGIME | VIX={vix:.1f} ≥ {VIX_STRESS}] {sig.reason}"
                )
            return sig

        # ── 3. Transition / Hybrid Regime (18.0 ≤ VIX < 25.0) ─────────
        mr_sig  = self.mr_scorer.score(row)
        mom_sig = self.mom_scorer.score(row, df_slice)

        mr_active  = mr_sig.is_tradeable()
        mom_active = mom_sig.is_tradeable()

        # Case 3a: Both inactive
        if not mr_active and not mom_active:
            return self._no_trade(row, price, f"[HYBRID REGIME | VIX={vix:.1f}] Neither scorer triggered")

        # Case 3b: Both active and agree on direction -> Strong confirmation
        if mr_active and mom_active and mr_sig.direction == mom_sig.direction:
            # Blend probability (50/50 weight in transition zone)
            blended_prob = round((mr_sig.probability + mom_sig.probability) / 2.0, 1)
            # Take SL/TP from the more conservative (closer SL) signal for safety in transition
            chosen_sig = mr_sig if abs(mr_sig.entry_price - mr_sig.stop_loss) < abs(mom_sig.entry_price - mom_sig.stop_loss) else mom_sig
            return replace(
                chosen_sig,
                probability = blended_prob,
                confidence  = "HIGH" if blended_prob >= 75 else "MEDIUM",
                reason      = f"[HYBRID AGREE | VIX={vix:.1f}] MR ({mr_sig.probability}%) + MOM ({mom_sig.probability}%) agreed on {mr_sig.direction} | {chosen_sig.reason}"
            )

        # Case 3c: Disagreement or only one active -> Take stronger signal with 0.85x dampener
        active_sig = mr_sig if mr_active else mom_sig
        if mr_active and mom_active:
            # If they disagree, prefer the one with higher distance from threshold (stronger confidence)
            mr_strength  = mr_sig.probability if mr_sig.direction == "LONG" else (100.0 - mr_sig.probability)
            mom_strength = mom_sig.probability if mom_sig.direction == "LONG" else (100.0 - mom_sig.probability)
            active_sig = mr_sig if mr_strength > mom_strength else mom_sig
            source_label = "MR" if active_sig is mr_sig else "MOM"
            conflict_note = f" (overrode conflicting {'MOM' if source_label=='MR' else 'MR'})"
        else:
            source_label = "MR" if mr_active else "MOM"
            conflict_note = ""

        # Apply transition uncertainty dampener
        damped_prob = active_sig.probability * 0.85 if active_sig.direction == "LONG" else 100.0 - ((100.0 - active_sig.probability) * 0.85)
        damped_prob = round(max(0.0, min(100.0, damped_prob)), 1)

        # Re-check threshold after dampening
        check_pass = damped_prob >= self.threshold if active_sig.direction == "LONG" else damped_prob <= (100.0 - self.threshold)
        if not check_pass:
            return self._no_trade(
                row, price,
                f"[HYBRID DAMPED | VIX={vix:.1f}] {source_label} sig damped from {active_sig.probability}% to {damped_prob}% (< threshold)"
            )

        return replace(
            active_sig,
            probability = damped_prob,
            confidence  = "LOW" if damped_prob < 65 else "MEDIUM",
            reason      = f"[HYBRID {source_label} | VIX={vix:.1f}]{conflict_note} Damped prob: {damped_prob}% | {active_sig.reason}"
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
