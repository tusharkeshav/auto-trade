# ─────────────────────────────────────────────────────────────────
#  indicators/support_resistance.py
#
#  Detects meaningful price levels algorithmically:
#    1. Daily Pivot Points  — recalculate from previous day's OHLC
#    2. Swing Highs & Lows — real historical rejection zones
#    3. Clustering          — merges nearby levels to reduce noise
#    4. Proximity scoring   — per-row: how close is price to S/R?
#
#  Output columns feed directly into the probability engine:
#    sr_at_support       → price is within `zone_pct` of a support level
#    sr_at_resistance    → price is within `zone_pct` of a resistance level
#    sr_support_dist_pct → % distance below nearest support (0 = right at it)
#    sr_resist_dist_pct  → % distance above nearest resistance
#    pivot_p / pivot_r1 / pivot_r2 / pivot_s1 / pivot_s2
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations   # enables X | Y and list[X] on Python 3.9+

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────

def _detect_swing_highs(df: pd.DataFrame, window: int) -> list[float]:
    """
    A swing high is a candle whose HIGH is the highest point
    within `window` candles on each side.

    These represent real resistance zones — price tried to go higher
    and got rejected. Institutions place sell orders here.
    """
    highs = df["high"]
    # Rolling max over (2*window + 1) candles centered on each candle
    rolling_max = highs.rolling(window=2 * window + 1, center=False).max()
    swing_mask  = highs == rolling_max

    return highs[swing_mask].dropna().tolist()


def _detect_swing_lows(df: pd.DataFrame, window: int) -> list[float]:
    """
    A swing low is a candle whose LOW is the lowest point
    within `window` candles on each side.

    These are real support zones — price fell here and buyers stepped in.
    """
    lows = df["low"]
    rolling_min = lows.rolling(window=2 * window + 1, center=False).min()
    swing_mask  = lows == rolling_min

    return lows[swing_mask].dropna().tolist()


def _cluster_levels(levels: list[float], cluster_pct: float) -> list[float]:
    """
    Merge price levels that are within `cluster_pct`% of each other.

    Why: a swing high at $64,100 and a pivot R1 at $64,090 are
    effectively the same zone. Clustering combines them into one
    precise level (their average), reducing noise.

    Returns levels sorted ascending.
    """
    if not levels:
        return []

    levels = sorted(levels)
    clusters: list[list[float]] = [[levels[0]]]

    for level in levels[1:]:
        last_cluster_mean = np.mean(clusters[-1])
        pct_diff = abs(level - last_cluster_mean) / last_cluster_mean * 100

        if pct_diff <= cluster_pct:
            clusters[-1].append(level)      # merge into current cluster
        else:
            clusters.append([level])        # start a new cluster

    return [float(np.mean(c)) for c in clusters]


def _nearest_below(price: float, levels: list[float]) -> float | None:
    """Returns the highest level that is at or below `price`."""
    candidates = [l for l in levels if l <= price]
    return max(candidates) if candidates else None


def _nearest_above(price: float, levels: list[float]) -> float | None:
    """Returns the lowest level that is at or above `price`."""
    candidates = [l for l in levels if l >= price]
    return min(candidates) if candidates else None


# ─────────────────────────────────────────────────────────────────
#  1. Daily Pivot Points
# ─────────────────────────────────────────────────────────────────

def add_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classic daily pivot points — computed from the previous day's
    High, Low, and Close.  Resampled to daily and forward-filled
    so every hourly candle knows its current pivot levels.

    Adds columns:
      pivot_p  — the pivot (centre of gravity for the day)
      pivot_r1 — first resistance above pivot
      pivot_r2 — second resistance (stronger)
      pivot_s1 — first support below pivot
      pivot_s2 — second support (stronger)

    These levels are widely watched by institutions, which makes
    them self-fulfilling — price genuinely tends to react at them.
    """
    df = df.copy()

    # Resample to daily OHLC
    daily_high  = df["high"].resample("D").max()
    daily_low   = df["low"].resample("D").min()
    daily_close = df["close"].resample("D").last()

    # Shift by 1 day → use *yesterday's* values for today's pivots
    prev_high  = daily_high.shift(1)
    prev_low   = daily_low.shift(1)
    prev_close = daily_close.shift(1)

    pivot = (prev_high + prev_low + prev_close) / 3
    r1    = (2 * pivot) - prev_low
    r2    = pivot + (prev_high - prev_low)
    s1    = (2 * pivot) - prev_high
    s2    = pivot - (prev_high - prev_low)

    # Reindex back to the original hourly index and forward-fill
    def _ffill(series: pd.Series) -> pd.Series:
        return series.reindex(df.index, method="ffill")

    df["pivot_p"]  = _ffill(pivot).round(2)
    df["pivot_r1"] = _ffill(r1).round(2)
    df["pivot_r2"] = _ffill(r2).round(2)
    df["pivot_s1"] = _ffill(s1).round(2)
    df["pivot_s2"] = _ffill(s2).round(2)

    return df


# ─────────────────────────────────────────────────────────────────
#  2. Swing Levels + Proximity Scoring
# ─────────────────────────────────────────────────────────────────

def add_sr_proximity(
    df: pd.DataFrame,
    swing_window: int  = 5,      # candles on each side to qualify as a swing
    cluster_pct: float = 0.3,    # merge levels within 0.3% of each other
    max_levels: int    = 10,     # keep only the N most recent levels
    zone_pct: float    = 0.3,    # within 0.3% = "at" a S/R level
) -> pd.DataFrame:
    """
    Detect swing highs/lows, cluster them, then compute per-row
    proximity metrics vs the unified support/resistance level set.

    Adds columns:
      sr_support_price     — price of nearest support below current close
      sr_resist_price      — price of nearest resistance above current close
      sr_support_dist_pct  — % distance: (close − support) / close × 100
                             0.0 = price is right at support ← most bullish
      sr_resist_dist_pct   — % distance: (resistance − close) / close × 100
                             0.0 = price is right at resistance ← most bearish
      sr_at_support        — True when sr_support_dist_pct ≤ zone_pct
      sr_at_resistance     — True when sr_resist_dist_pct  ≤ zone_pct

    Probability engine usage:
      sr_at_support    → +probability for LONG  (price bouncing from floor)
      sr_at_resistance → −probability for LONG  (price hitting a ceiling)
      sr_support_dist_pct → continuous weight (closer = stronger signal)
    """
    df = df.copy()

    # ── Build unified level set ───────────────────────────────────
    swing_highs = _detect_swing_highs(df, swing_window)
    swing_lows  = _detect_swing_lows(df, swing_window)

    # Only keep the most recent levels (older ones are less relevant)
    swing_highs = sorted(swing_highs)[-max_levels:]
    swing_lows  = sorted(swing_lows)[:max_levels]         # keep lowest N as supports

    # Cluster to remove near-duplicate levels
    resistance_levels = _cluster_levels(swing_highs, cluster_pct)
    support_levels    = _cluster_levels(swing_lows,  cluster_pct)

    # Merge supports + resistances into one unified list
    # (a former resistance once broken becomes support, and vice versa)
    all_levels = _cluster_levels(resistance_levels + support_levels, cluster_pct)

    # ── Per-row proximity computation ─────────────────────────────
    support_prices:     list[float | None] = []
    resist_prices:      list[float | None] = []
    support_dist_pcts:  list[float | None] = []
    resist_dist_pcts:   list[float | None] = []

    for price in df["close"]:
        sup = _nearest_below(price, all_levels)
        res = _nearest_above(price, all_levels)

        s_dist = round((price - sup) / price * 100, 4) if sup else None
        r_dist = round((res - price) / price * 100, 4) if res else None

        support_prices.append(sup)
        resist_prices.append(res)
        support_dist_pcts.append(s_dist)
        resist_dist_pcts.append(r_dist)

    df["sr_support_price"]    = support_prices
    df["sr_resist_price"]     = resist_prices
    df["sr_support_dist_pct"] = support_dist_pcts
    df["sr_resist_dist_pct"]  = resist_dist_pcts

    # Boolean flags for the probability engine
    df["sr_at_support"]    = df["sr_support_dist_pct"].apply(
        lambda x: x is not None and x <= zone_pct
    )
    df["sr_at_resistance"] = df["sr_resist_dist_pct"].apply(
        lambda x: x is not None and x <= zone_pct
    )

    return df


# ─────────────────────────────────────────────────────────────────
#  Public entry point — call both in one shot
# ─────────────────────────────────────────────────────────────────

def add_support_resistance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all support & resistance metrics in one call.

    Combines:
      - Daily pivot points (institutional levels)
      - Swing high/low detection + clustering (historical zones)
      - Per-row proximity scoring (feeds probability engine)
    """
    df = add_pivot_points(df)
    df = add_sr_proximity(df)
    return df
