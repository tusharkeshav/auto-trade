# ─────────────────────────────────────────────────────────────────
#  probability/india_signal_scorer.py
#  NIFTY / BANKNIFTY signal scorer.
#
#  Inherits the 3-gate probability engine from SignalScorer but
#  overrides the macro-shield with IST session filter + VIX layer.
#  Core Gate A/B/C logic (RSI, BB, S/R, MACD, Stoch) unchanged.
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from datetime import datetime
from zoneinfo  import ZoneInfo

import pandas as pd

from probability.signal_scorer import SignalScorer, TradeSignal, _nan, _calculate_risk_levels
from engine.india_regime       import detect_india_regime, IndiaRegimeDecision
from config.india_settings     import (
    INDIA_MACRO_MAX_ADX,
    NSE_MACRO_START_HOUR_IST, NSE_MACRO_START_MIN_IST,
    NSE_MACRO_END_HOUR_IST,   NSE_MACRO_END_MIN_IST,
    INDIA_SIGNAL_THRESHOLD,
    VIX_NORMAL_HIGH,
)

IST = ZoneInfo("Asia/Kolkata")


class IndiaSignalScorer(SignalScorer):
    """
    Mean-reversion signal scorer for NIFTY/BANKNIFTY.

    Changes vs BTC SignalScorer:
      - IST session gate replaces UTC 16-24 gate
      - India VIX filter replaces Monday/Tuesday exclusion
      - signal_multiplier from IndiaRegimeDecision applied to raw score
      - Prices in ₹ (no dollar signs in logs)
    """

    def __init__(
        self,
        symbol:           str   = "NIFTY50",
        long_threshold:   float = INDIA_SIGNAL_THRESHOLD,
        short_threshold:  float = 100 - INDIA_SIGNAL_THRESHOLD,
        current_vix:      float = 15.0,
        interval:         str   = "15m",
    ):
        super().__init__(symbol, long_threshold, short_threshold)
        self.current_vix = current_vix
        self._is_intraday = interval not in ("1d", "1wk", "1mo")

    def score(self, row: pd.Series) -> TradeSignal:
        price = row["close"]
        adx   = row.get("adx", float("nan"))

        # ── Macro Shield Gate 1: ADX ──────────────────────────────
        if _nan(adx) or adx > INDIA_MACRO_MAX_ADX:
            return self._no_trade(row, price, f"ADX={adx:.1f} > {INDIA_MACRO_MAX_ADX} (trending)")

        # ── Macro Shield Gate 2: IST session window (intraday only) ──
        if self._is_intraday and hasattr(row, "name") and row.name is not None:
            ts_ist = _to_ist(row.name)
            if ts_ist is not None:
                session_start = ts_ist.replace(
                    hour=NSE_MACRO_START_HOUR_IST,
                    minute=NSE_MACRO_START_MIN_IST,
                    second=0, microsecond=0,
                )
                session_end = ts_ist.replace(
                    hour=NSE_MACRO_END_HOUR_IST,
                    minute=NSE_MACRO_END_MIN_IST,
                    second=0, microsecond=0,
                )
                if not (session_start <= ts_ist <= session_end):
                    return self._no_trade(
                        row, price,
                        f"IST={ts_ist.strftime('%H:%M')} outside session "
                        f"[{NSE_MACRO_START_HOUR_IST:02d}:{NSE_MACRO_START_MIN_IST:02d}"
                        f"–{NSE_MACRO_END_HOUR_IST:02d}:{NSE_MACRO_END_MIN_IST:02d} IST]"
                    )

        # ── India VIX + ADX regime ────────────────────────────────
        regime: IndiaRegimeDecision = detect_india_regime(row, self.current_vix)

        if regime.signal_multiplier == 0.0:
            return self._no_trade(row, price, f"Regime blocked: {regime.reason}")

        if regime.regime != "MEAN_REVERSION":
            return self._no_trade(row, price, f"Regime={regime.regime} — mean-rev scorer inactive")

        # ── 3-Gate scoring (inherited logic) ──────────────────────
        # Call _score_core() to skip parent's BTC macro shields (UTC session / weekday).
        base_signal = self._score_core(row)

        if not base_signal.is_tradeable():
            return base_signal   # NO_TRADE from gate A/B/C — keep as-is

        # Apply VIX multiplier: scale raw probability UP (low VIX) or DOWN (elevated VIX)
        adjusted_prob = base_signal.probability * regime.signal_multiplier
        adjusted_prob = max(0.0, min(100.0, adjusted_prob))

        # Re-check threshold after VIX adjustment
        if base_signal.direction == "LONG" and adjusted_prob < self.long_threshold:
            return self._no_trade(
                row, price,
                f"VIX-adjusted prob {adjusted_prob:.1f}% < threshold {self.long_threshold} "
                f"(VIX={self.current_vix:.1f}, zone={regime.vix_zone})"
            )
        if base_signal.direction == "SHORT":
            # SHORT probability is inverted (100-score), so strong short = low number
            # VIX multiplier: adjust the raw score (not inverted prob)
            raw_short_score = 100.0 - base_signal.probability
            adjusted_score  = raw_short_score * regime.signal_multiplier
            adjusted_score  = max(0.0, min(100.0, adjusted_score))
            adjusted_prob   = 100.0 - adjusted_score
            if adjusted_score < self.long_threshold:
                return self._no_trade(
                    row, price,
                    f"VIX-adjusted short score {adjusted_score:.1f}% < threshold {self.long_threshold} "
                    f"(VIX={self.current_vix:.1f}, zone={regime.vix_zone})"
                )

        # Return adjusted signal (same prices, updated probability)
        from dataclasses import replace
        return replace(
            base_signal,
            probability=round(adjusted_prob, 2),
            reason=f"{base_signal.reason} | VIX={self.current_vix:.1f} ({regime.vix_zone}) × {regime.signal_multiplier:.2f}",
        )

    # ── Helpers ───────────────────────────────────────────────────

    def _no_trade(self, row: pd.Series, price: float, reason: str) -> TradeSignal:
        return TradeSignal(
            symbol=self.symbol,
            timestamp=getattr(row, "name", None),
            direction="NO_TRADE",
            probability=50.0,
            confidence="LOW",
            raw_score=0.0,
            entry_price=price,
            stop_loss=price,
            take_profit1=price,
            take_profit2=price,
            risk_amount=0.0,
            reason=reason,
        )


def _to_ist(ts) -> datetime | None:
    """Convert a pandas Timestamp (possibly UTC-aware) to IST datetime."""
    try:
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            return ts.to_pydatetime().astimezone(IST)
        # Naive: assume UTC
        from datetime import timezone
        return ts.to_pydatetime().replace(tzinfo=timezone.utc).astimezone(IST)
    except Exception:
        return None
