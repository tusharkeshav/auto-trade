# ─────────────────────────────────────────────────────────────────
#  probability/momentum_scorer.py
#
#  Momentum Probability Engine (3-Layer Gated Architecture).
#
#  Takes a single row of indicator values and produces a trade signal
#  based on TRENDING conditions (opposite of mean reversion).
#
#  Gates:
#    - Gate A: Trend Structure (Max 35) - Is the trend established?
#    - Gate B: Momentum Confirmation (Max 40) - Is momentum accelerating?
#    - Gate C: Entry Timing (Max 25) - Is this a good entry point?
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from config.profiles import get_profile
from config.settings import MACRO_SESSION_START, MACRO_SESSION_END


# ─────────────────────────────────────────────────────────────────
#  Trade Signal (reuses same structure as mean-rev scorer)
# ─────────────────────────────────────────────────────────────────

from probability.signal_scorer import TradeSignal


def _nan(*values) -> bool:
    """True if any value is NaN or None."""
    return any(v is None or (isinstance(v, float) and math.isnan(v)) for v in values)


def _calculate_risk_levels(
    row: pd.Series,
    direction: str,
    atr_sl_mult: float = 2.0,
    atr_tp_ratio: float = 3.0,
) -> tuple[float, float, float, float]:
    """Returns (stop_loss, take_profit1, take_profit2, risk_amount).

    Momentum needs wider SL/TP than mean reversion:
      - SL at 2.0x ATR (vs 1.25x for mean rev)
      - TP at 3.0x risk (vs 1.5x for mean rev)
    """
    entry = row["close"]
    atr   = row["atr"] if not _nan(row["atr"]) else entry * 0.004

    if direction == "LONG":
        stop_loss    = round(entry - (atr_sl_mult * atr), 2)
        risk_amount  = entry - stop_loss
        take_profit1 = round(entry + (atr_tp_ratio * risk_amount), 2)
        take_profit2 = take_profit1  # All-in, All-out
    else:
        stop_loss    = round(entry + (atr_sl_mult * atr), 2)
        risk_amount  = stop_loss - entry
        take_profit1 = round(entry - (atr_tp_ratio * risk_amount), 2)
        take_profit2 = take_profit1

    return stop_loss, take_profit1, take_profit2, round(risk_amount, 2)


# ─────────────────────────────────────────────────────────────────
#  The Momentum Probability Engine
# ─────────────────────────────────────────────────────────────────

class MomentumScorer:
    """
    Scores a single market snapshot using a 3-Gate momentum architecture.

    Fires when:
      - Price is trending (EMA aligned, DMI confirming)
      - Momentum is accelerating (RSI, MACD, volume)
      - Entry timing is right (breakout or pullback to trend)
    """

    def __init__(
        self,
        symbol:           str   = "BTCUSDT",
        long_threshold:   float = 60.0,
        short_threshold:  float = 35.0,
        momentum_adx_min: float = 20.0,
        atr_sl_mult:      float = 2.0,
        atr_tp_ratio:     float = 3.0,
    ):
        self.symbol           = symbol
        self.long_threshold   = long_threshold
        self.short_threshold  = short_threshold
        self.momentum_adx_min = momentum_adx_min
        self.atr_sl_mult      = atr_sl_mult
        self.atr_tp_ratio     = atr_tp_ratio
        self.profile          = get_profile(symbol)

    def score(self, row: pd.Series) -> TradeSignal:
        price = row["close"]

        # ─────────────────────────────────────────────────────────
        # MACRO SHIELD: Must be in trending regime
        # ─────────────────────────────────────────────────────────
        adx = row.get("adx", float('nan'))
        if _nan(adx) or adx < self.momentum_adx_min:
            return TradeSignal(
                self.symbol, getattr(row, 'name', None), "NO_TRADE",
                50.0, "LOW", 0.0, price, price, price, price, 0.0,
                reason=f"ADX={adx:.1f} < {self.momentum_adx_min} (not trending enough for momentum)"
            )

        # Session filter (same as mean reversion — liquidity window)
        if hasattr(row, "name") and row.name is not None:
            hour = row.name.hour
            if not (MACRO_SESSION_START <= hour < MACRO_SESSION_END):
                return TradeSignal(
                    self.symbol, getattr(row, 'name', None), "NO_TRADE",
                    50.0, "LOW", 0.0, price, price, price, price, 0.0,
                    reason=f"UTC hour={hour} outside session [{MACRO_SESSION_START}-{MACRO_SESSION_END}]"
                )

        # Day-of-week filter (same as mean reversion — Mon+Tue underperform)
        if hasattr(row, "name") and row.name is not None:
            weekday = row.name.weekday()
            if weekday in (0, 1):
                return TradeSignal(
                    self.symbol, getattr(row, 'name', None), "NO_TRADE",
                    50.0, "LOW", 0.0, price, price, price, price, 0.0,
                    reason=f"Weekday={weekday} (Mon/Tue excluded)"
                )

        # ─────────────────────────────────────────────────────────
        # GATE A: Trend Structure (Max 35)
        # ─────────────────────────────────────────────────────────
        gate_a_long, gate_a_short = 0.0, 0.0

        # 1. EMA Alignment (Max 15)
        ema12 = row.get("ema_12", float('nan'))
        ema26 = row.get("ema_26", float('nan'))
        sma50 = row.get("sma_50", float('nan'))

        if not _nan(ema12, ema26, sma50):
            if ema12 > ema26 > sma50:
                gate_a_long += 15.0
            elif ema12 < ema26 < sma50:
                gate_a_short += 15.0
            elif ema12 > ema26:
                gate_a_long += 8.0   # partial alignment
            elif ema12 < ema26:
                gate_a_short += 8.0

        # 2. DMI Confirmation (Max 12)
        di_plus  = row.get("di_plus", float('nan'))
        di_minus = row.get("di_minus", float('nan'))

        if not _nan(di_plus, di_minus):
            if di_plus > di_minus:
                gate_a_long += 12.0
            elif di_minus > di_plus:
                gate_a_short += 12.0

        # 3. Price vs SMA200 (Max 8)
        sma200 = row.get("sma_200", float('nan'))
        if not _nan(sma200):
            if price > sma200:
                gate_a_long += 8.0
            elif price < sma200:
                gate_a_short += 8.0

        # ─────────────────────────────────────────────────────────
        # GATE B: Momentum Confirmation (Max 40)
        # ─────────────────────────────────────────────────────────
        gate_b_long, gate_b_short = 0.0, 0.0

        # 1. RSI in momentum zone (Max 12)
        rsi = row.get("rsi", float('nan'))
        if not _nan(rsi):
            if 50 <= rsi <= 70:
                gate_b_long += 12.0
            elif 30 <= rsi <= 50:
                gate_b_short += 12.0
            elif rsi > 70:
                gate_b_long += 5.0   # strong but extended
            elif rsi < 30:
                gate_b_short += 5.0

        # 2. MACD rising (Max 10)
        macd_hist      = row.get("macd_hist", float('nan'))
        macd_prev_hist = row.get("macd_hist_prev", float('nan'))

        if not _nan(macd_hist):
            # Bullish: positive & rising
            if macd_hist > 0:
                gate_b_long += 10.0
            # Bearish: negative & falling
            elif macd_hist < 0:
                gate_b_short += 10.0

        # 3. Volume confirmation (Max 10)
        volume_ratio = row.get("volume_ratio", float('nan'))
        if not _nan(volume_ratio):
            if volume_ratio >= 1.5:
                gate_b_long += 10.0
                gate_b_short += 10.0
            elif volume_ratio >= 1.0:
                gate_b_long += 5.0
                gate_b_short += 5.0

        # 4. ATR expansion (Max 8)
        atr = row.get("atr", float('nan'))
        atr_prev = row.get("atr_prev", float('nan'))
        if not _nan(atr, atr_prev) and atr_prev > 0:
            if atr > atr_prev * 1.1:  # ATR expanding >10%
                gate_b_long += 8.0
                gate_b_short += 8.0

        # ─────────────────────────────────────────────────────────
        # GATE C: Entry Timing (Max 25)
        # ─────────────────────────────────────────────────────────
        gate_c_long, gate_c_short = 0.0, 0.0

        # 1. Breakout from range (Max 12)
        high_20 = row.get("high_20", float('nan'))
        low_20  = row.get("low_20", float('nan'))
        if not _nan(high_20, low_20):
            if price >= high_20:
                gate_c_long += 12.0
            elif price <= low_20:
                gate_c_short += 12.0
            # Near breakout
            elif price >= high_20 * 0.98:
                gate_c_long += 6.0
            elif price <= low_20 * 1.02:
                gate_c_short += 6.0

        # 2. Pullback within trend (Max 8)
        sma_20 = row.get("sma_20", float('nan'))
        if not _nan(sma_20):
            # Price near SMA20 in uptrend = pullback entry
            ema12_ok = not _nan(ema12)
            if ema12_ok and ema12 > sma_20 and price <= sma_20 * 1.02:
                gate_c_long += 8.0
            elif ema12_ok and ema12 < sma_20 and price >= sma_20 * 0.98:
                gate_c_short += 8.0

        # 3. Consecutive momentum (Max 5)
        bull_count = row.get("bullish_candle_count", 0)
        bear_count = row.get("bearish_candle_count", 0)
        if bull_count >= 3:
            gate_c_long += 5.0
        if bear_count >= 3:
            gate_c_short += 5.0

        # ─────────────────────────────────────────────────────────
        # Final Processing
        # ─────────────────────────────────────────────────────────
        total_long  = gate_a_long + gate_b_long + gate_c_long
        total_short = gate_a_short + gate_b_short + gate_c_short

        if total_long >= total_short:
            raw_score = total_long
            direction_bias = "LONG"
            mapped_probability = total_long
            breakdown = [
                {"name": "Gate A: Trend Structure", "score": round(gate_a_long, 1), "max": 35},
                {"name": "Gate B: Momentum Conf",   "score": round(gate_b_long, 1), "max": 40},
                {"name": "Gate C: Entry Timing",    "score": round(gate_c_long, 1), "max": 25},
            ]
        else:
            raw_score = total_short
            direction_bias = "SHORT"
            mapped_probability = 100.0 - total_short
            breakdown = [
                {"name": "Gate A: Trend Structure", "score": round(gate_a_short, 1), "max": 35},
                {"name": "Gate B: Momentum Conf",   "score": round(gate_b_short, 1), "max": 40},
                {"name": "Gate C: Entry Timing",    "score": round(gate_c_short, 1), "max": 25},
            ]

        if mapped_probability >= self.long_threshold and direction_bias == "LONG":
            direction = "LONG"
        elif mapped_probability <= self.short_threshold and direction_bias == "SHORT":
            direction = "SHORT"
        else:
            direction = "NO_TRADE"

        if raw_score >= 70: confidence = "HIGH"
        elif raw_score >= 55: confidence = "MEDIUM"
        else: confidence = "LOW"

        highest_gate = max(breakdown, key=lambda x: x["score"])
        reason = f"Momentum {direction_bias} via {highest_gate['name']} ({highest_gate['score']} pts)"

        sl, tp1, tp2, risk = _calculate_risk_levels(
                row,
                direction if direction != "NO_TRADE" else "LONG",
                atr_sl_mult=self.atr_sl_mult,
                atr_tp_ratio=self.atr_tp_ratio,
            )

        return TradeSignal(
            symbol       = self.symbol,
            timestamp    = row.name if hasattr(row, "name") else datetime.utcnow(),
            direction    = direction,
            probability  = round(mapped_probability, 1),
            confidence   = confidence,
            raw_score    = round(raw_score, 1),
            entry_price  = round(price, 2),
            stop_loss    = sl,
            take_profit1 = tp1,
            take_profit2 = tp2,
            risk_amount  = risk,
            breakdown    = breakdown,
            reason       = reason,
        )
