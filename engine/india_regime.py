# ─────────────────────────────────────────────────────────────────
#  engine/india_regime.py
#  Regime detector for Indian indices (NIFTY/BANKNIFTY).
#
#  Extends the base ADX+BB regime detector with a third filter:
#  India VIX — the NSE's fear index for NIFTY 50.
#
#  Decision hierarchy:
#    1. VIX > VIX_STRESS (25)       → NO_TRADE (stress regime)
#    2. VIX in elevated zone (18-25) → require stronger signal
#    3. ADX / BB width               → MEAN_REVERSION vs MOMENTUM (same as BTC)
#    4. VIX < VIX_LOW (12)           → bonus: premium mean-rev zone
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from config.india_settings import (
    VIX_LOW, VIX_NORMAL_HIGH, VIX_STRESS,
    INDIA_REGIME_ADX_LOW, INDIA_REGIME_ADX_HIGH, INDIA_REGIME_BB_WIDTH_THRESH,
)


@dataclass
class IndiaRegimeDecision:
    """
    Output of the India-specific regime detector.
    Superset of RegimeDecision with VIX layer added.
    """
    regime:         str    # "MEAN_REVERSION" | "MOMENTUM" | "NO_TRADE"
    reason:         str
    adx:            float
    bb_width_pctile: float
    vix:            float
    vix_zone:       str    # "LOW" | "NORMAL" | "ELEVATED" | "STRESS"
    signal_multiplier: float   # 1.0 = normal, < 1.0 = require higher score, 0.0 = blocked


def vix_zone(vix: float) -> str:
    """Classify India VIX level into a named zone."""
    if math.isnan(vix):
        return "UNKNOWN"
    if vix > VIX_STRESS:
        return "STRESS"
    if vix > VIX_NORMAL_HIGH:
        return "ELEVATED"
    if vix < VIX_LOW:
        return "LOW"
    return "NORMAL"


def detect_india_regime(row: pd.Series, current_vix: float) -> IndiaRegimeDecision:
    """
    Detect regime for NIFTY/BANKNIFTY with VIX filter layered on top.

    Args:
        row:         DataFrame row with 'adx' and optionally 'bb_width_percentile'.
        current_vix: Latest India VIX value (fetched once per loop iteration).

    Returns:
        IndiaRegimeDecision.
    """
    adx            = row.get("adx", float("nan"))
    bb_width_pctile = row.get("bb_width_percentile", float("nan"))
    zone           = vix_zone(current_vix)

    # ── VIX stress: block all mean-reversion longs ─────────────
    if zone == "STRESS":
        return IndiaRegimeDecision(
            regime="NO_TRADE",
            reason=f"India VIX={current_vix:.1f} > {VIX_STRESS} (STRESS) — mean-rev blocked",
            adx=adx, bb_width_pctile=bb_width_pctile,
            vix=current_vix, vix_zone=zone,
            signal_multiplier=0.0,
        )

    # ── VIX elevated: require 20% stronger signal ───────────────
    if zone == "ELEVATED":
        multiplier = 0.80   # raise effective threshold by ~25% (score × 0.8 = same as needing higher raw score)
    elif zone == "LOW":
        multiplier = 1.10   # premium mean-rev zone — slightly easier threshold
    else:
        multiplier = 1.00   # NORMAL

    # ── ADX NaN guard ───────────────────────────────────────────
    if math.isnan(adx):
        return IndiaRegimeDecision(
            regime="NO_TRADE",
            reason="ADX is NaN",
            adx=float("nan"), bb_width_pctile=bb_width_pctile,
            vix=current_vix, vix_zone=zone,
            signal_multiplier=0.0,
        )

    # ── Strong mean-reversion zone ─────────────────────────────
    if adx <= INDIA_REGIME_ADX_LOW:
        return IndiaRegimeDecision(
            regime="MEAN_REVERSION",
            reason=f"ADX={adx:.1f} ≤ {INDIA_REGIME_ADX_LOW} | VIX={current_vix:.1f} ({zone})",
            adx=adx, bb_width_pctile=bb_width_pctile,
            vix=current_vix, vix_zone=zone,
            signal_multiplier=multiplier,
        )

    # ── Strong momentum zone ────────────────────────────────────
    if adx >= INDIA_REGIME_ADX_HIGH:
        return IndiaRegimeDecision(
            regime="MOMENTUM",
            reason=f"ADX={adx:.1f} ≥ {INDIA_REGIME_ADX_HIGH} | VIX={current_vix:.1f} ({zone})",
            adx=adx, bb_width_pctile=bb_width_pctile,
            vix=current_vix, vix_zone=zone,
            signal_multiplier=multiplier,
        )

    # ── Hybrid zone (ADX 20-30): use BB width ──────────────────
    if not math.isnan(bb_width_pctile) and bb_width_pctile > INDIA_REGIME_BB_WIDTH_THRESH:
        return IndiaRegimeDecision(
            regime="MOMENTUM",
            reason=f"ADX={adx:.1f} hybrid + BB width {bb_width_pctile:.2f} > {INDIA_REGIME_BB_WIDTH_THRESH} | VIX={current_vix:.1f} ({zone})",
            adx=adx, bb_width_pctile=bb_width_pctile,
            vix=current_vix, vix_zone=zone,
            signal_multiplier=multiplier,
        )

    return IndiaRegimeDecision(
        regime="MEAN_REVERSION",
        reason=f"ADX={adx:.1f} hybrid + BB width {bb_width_pctile:.2f} ≤ {INDIA_REGIME_BB_WIDTH_THRESH} | VIX={current_vix:.1f} ({zone})",
        adx=adx, bb_width_pctile=bb_width_pctile,
        vix=current_vix, vix_zone=zone,
        signal_multiplier=multiplier,
    )
