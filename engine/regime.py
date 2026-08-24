# ─────────────────────────────────────────────────────────────────
#  engine/regime.py
#
#  Regime Detector — decides whether market is in mean-reversion
#  or trending/momentum regime based on ADX + BB width.
#
#  ADX ≤ 20  → Mean reversion zone
#  ADX ≥ 30  → Momentum/trending zone
#  20-30     → Hybrid: check BB width percentile
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations
from dataclasses import dataclass

import pandas as pd

from config.settings import REGIME_ADX_LOW, REGIME_ADX_HIGH, REGIME_BB_WIDTH_THRESH

ADX_LOW         = REGIME_ADX_LOW
ADX_HIGH        = REGIME_ADX_HIGH
BB_WIDTH_THRESH = REGIME_BB_WIDTH_THRESH


@dataclass
class RegimeDecision:
    """
    Output of the regime detector for one candle.
    """
    regime: str          # "MEAN_REVERSION" | "MOMENTUM" | "NO_TRADE"
    reason: str          # Human-readable explanation
    adx: float           # Current ADX value
    bb_width_pctile: float  # BB width percentile


def detect_regime(row: pd.Series) -> RegimeDecision:
    """
    Detect which regime the market is in based on a single row of indicators.

    Args:
        row: DataFrame row with at minimum 'adx' column.
             If 'bb_width_percentile' is available, uses it for hybrid zone.

    Returns:
        RegimeDecision with the detected regime and reason.
    """
    import math

    adx = row.get("adx", float('nan'))
    if math.isnan(adx):
        return RegimeDecision("NO_TRADE", "ADX is NaN", float('nan'), float('nan'))

    bb_width_pctile = row.get("bb_width_percentile", float('nan'))

    # ── Strong mean reversion zone ──────────────────────────────
    if adx <= ADX_LOW:
        return RegimeDecision(
            "MEAN_REVERSION",
            f"ADX={adx:.1f} ≤ {ADX_LOW} (low trend — mean reversion)",
            adx, bb_width_pctile
        )

    # ── Strong momentum zone ────────────────────────────────────
    if adx >= ADX_HIGH:
        return RegimeDecision(
            "MOMENTUM",
            f"ADX={adx:.1f} ≥ {ADX_HIGH} (strong trend — momentum)",
            adx, bb_width_pctile
        )

    # ── Hybrid zone: ADX 20-30 ──────────────────────────────────
    # Use BB width to decide: low vol → mean rev, high vol → momentum
    if not math.isnan(bb_width_pctile) and bb_width_pctile > BB_WIDTH_THRESH:
        return RegimeDecision(
            "MOMENTUM",
            f"ADX={adx:.1f} hybrid + BB width {bb_width_pctile:.2f} > {BB_WIDTH_THRESH} (expansion → momentum)",
            adx, bb_width_pctile
        )

    return RegimeDecision(
        "MEAN_REVERSION",
        f"ADX={adx:.1f} hybrid + BB width {bb_width_pctile:.2f} ≤ {BB_WIDTH_THRESH} (compression → mean rev)",
        adx, bb_width_pctile
    )
