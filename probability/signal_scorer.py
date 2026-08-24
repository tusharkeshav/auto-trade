# ─────────────────────────────────────────────────────────────────
#  probability/signal_scorer.py
#
#  The Probability Engine (3-Layer Gated Architecture).
#
#  Takes a single row of indicator values and produces a trade signal.
#  Uses a 3-Gate system to avoid trend/momentum cancellation:
#    - Gate A: Location (Max 40) - Are we at a key level?
#    - Gate B: Confirmation (Max 35) - Is momentum reversing?
#    - Gate C: Context (Max 25) - Is macro trend helping?
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from config.profiles  import get_profile
from config.settings  import MACRO_MAX_ADX, MACRO_SESSION_START, MACRO_SESSION_END


# ─────────────────────────────────────────────────────────────────
#  Trade Signal — the output of the engine
# ─────────────────────────────────────────────────────────────────

@dataclass
class TradeSignal:
    """
    Complete trade recommendation produced by the probability engine.

    Risk levels follow the user's conservative strategy:
      - Stop loss  : 1.0 × ATR *below* nearest support
      - Take profit 1: entry + 1.0 × risk  → book 50% here
      - Take profit 2: entry + 1.5 × risk  → let remaining 50% run
    """
    symbol:      str
    timestamp:   datetime

    direction:   str      # "LONG" | "SHORT" | "NO_TRADE"
    probability: float    # 0–100  (≥70 = trade, ≤30 = short, else skip)
    confidence:  str      # "HIGH" | "MEDIUM" | "LOW"
    raw_score:   float    # the winning unmapped score (0-100)

    entry_price: float
    stop_loss:   float    
    take_profit1: float   
    take_profit2: float   
    risk_amount:  float   

    breakdown:   list[dict] = field(default_factory=list)
    reason:      str = ""   

    def is_tradeable(self) -> bool:
        return self.direction in ("LONG", "SHORT")

    def __str__(self) -> str:
        if not self.is_tradeable():
            return (
                f"[{self.symbol}] NO TRADE  |  "
                f"Probability: {self.probability:.1f}%  |  "
                f"Reason: {self.reason}"
            )
        return (
            f"[{self.symbol}] {self.direction}  |  "
            f"Probability: {self.probability:.1f}%  ({self.confidence})  |  "
            f"Entry: ${self.entry_price:,.2f}  |  "
            f"SL: ${self.stop_loss:,.2f}  |  "
            f"TP1: ${self.take_profit1:,.2f}  |  "
            f"TP2: ${self.take_profit2:,.2f}"
        )


def _nan(*values) -> bool:
    """True if any value is NaN or None."""
    return any(v is None or (isinstance(v, float) and math.isnan(v)) for v in values)


# ─────────────────────────────────────────────────────────────────
#  Risk level calculator
# ─────────────────────────────────────────────────────────────────

def _calculate_risk_levels(
    row: pd.Series,
    direction: str,
) -> tuple[float, float, float, float]:
    """Returns (stop_loss, take_profit1, take_profit2, risk_amount)."""
    entry = row["close"]
    atr   = row["atr"] if not _nan(row["atr"]) else entry * 0.004

    sup = row["sr_support_price"]
    res = row["sr_resist_price"]

    if direction == "LONG":
        base_support = sup if not _nan(sup) else (entry - atr)
        stop_loss    = base_support - (1.0 * atr)

        risk_amount  = entry - stop_loss
        take_profit1 = entry + (1.0 * risk_amount)   
        take_profit2 = entry + (1.5 * risk_amount)   

    else:  # SHORT
        base_resist  = res if not _nan(res) else (entry + atr)
        stop_loss    = base_resist + (1.0 * atr)

        risk_amount  = stop_loss - entry
        take_profit1 = entry - (1.0 * risk_amount)   
        take_profit2 = entry - (1.5 * risk_amount)   

    return round(stop_loss, 2), round(take_profit1, 2), round(take_profit2, 2), round(risk_amount, 2)


# ─────────────────────────────────────────────────────────────────
#  The Probability Engine
# ─────────────────────────────────────────────────────────────────

class SignalScorer:
    """
    Scores a single market snapshot using the 3-Layer Gated Architecture.
    """

    def __init__(
        self,
        symbol:           str   = "BTCUSDT",
        long_threshold:   float = 70.0,   
        short_threshold:  float = 30.0,   
    ):
        self.symbol          = symbol
        self.long_threshold  = long_threshold
        self.short_threshold = short_threshold
        self.profile         = get_profile(symbol)

    def score(self, row: pd.Series) -> TradeSignal:
        price = row["close"]
        atr   = row["atr"] if not _nan(row["atr"]) else price * 0.004

        # ─────────────────────────────────────────────────────────
        # MACRO SHIELD
        # Gate 1: Trend regime — no trades during strong trends
        # ─────────────────────────────────────────────────────────
        adx = row.get("adx", float('nan'))
        if _nan(adx) or adx > MACRO_MAX_ADX:
            return TradeSignal(
                self.symbol, getattr(row, 'name', None), "NO_TRADE",
                50.0, "LOW", 0.0, price, price, price, price, 0.0,
                reason=f"ADX={adx:.1f} > {MACRO_MAX_ADX} (trending — skip)"
            )

        # Gate 2: Session filter — only trade during liquidity exhaustion window
        if hasattr(row, "name") and row.name is not None:
            hour = row.name.hour
            if not (MACRO_SESSION_START <= hour < MACRO_SESSION_END):
                return TradeSignal(
                    self.symbol, getattr(row, 'name', None), "NO_TRADE",
                    50.0, "LOW", 0.0, price, price, price, price, 0.0,
                    reason=f"UTC hour={hour} outside session [{MACRO_SESSION_START}-{MACRO_SESSION_END}]"
                )

        # Gate 3: Day-of-week filter — Mon (0) and Tue (1) underperform (PF<0.80)
        if hasattr(row, "name") and row.name is not None:
            weekday = row.name.weekday()
            if weekday in (0, 1):
                return TradeSignal(
                    self.symbol, getattr(row, 'name', None), "NO_TRADE",
                    50.0, "LOW", 0.0, price, price, price, price, 0.0,
                    reason=f"Weekday={weekday} (Mon/Tue excluded — PF<0.80)"
                )

        return self._score_core(row)

    def _score_core(self, row: pd.Series) -> TradeSignal:
        """Gate A/B/C scoring only — no macro shields. Used by India scorer."""
        price = row["close"]
        atr   = row["atr"] if not _nan(row["atr"]) else price * 0.004

        # ─────────────────────────────────────────────────────────
        # GATE A: Location (Max 40 points)
        # Are we at the right price?
        # ─────────────────────────────────────────────────────────
        gate_a_long, gate_a_short = 0.0, 0.0
        
        # 1. S/R Proximity (Max 20)
        sup_dist_abs = row["sr_support_dist_pct"] * price / 100 if not _nan(row["sr_support_dist_pct"]) else 9999
        res_dist_abs = row["sr_resist_dist_pct"] * price / 100 if not _nan(row["sr_resist_dist_pct"]) else 9999

        if sup_dist_abs <= atr * self.profile.sr_dist_near: gate_a_long += 20.0
        elif sup_dist_abs <= atr * self.profile.sr_dist_mid: gate_a_long += 12.0
        elif sup_dist_abs <= atr * self.profile.sr_dist_far: gate_a_long += 5.0

        if res_dist_abs <= atr * self.profile.sr_dist_near: gate_a_short += 20.0
        elif res_dist_abs <= atr * self.profile.sr_dist_mid: gate_a_short += 12.0
        elif res_dist_abs <= atr * self.profile.sr_dist_far: gate_a_short += 5.0

        # 2. Bollinger Bands (Max 12)
        if not _nan(row["bb_pct"]):
            if row["bb_pct"] <= self.profile.bb_lower_extreme: gate_a_long += 12.0
            elif row["bb_pct"] <= self.profile.bb_lower_mid: gate_a_long += 6.0
            if row["bb_pct"] >= self.profile.bb_upper_extreme: gate_a_short += 12.0
            elif row["bb_pct"] >= self.profile.bb_upper_mid: gate_a_short += 6.0

        # 3. Pivot Points (Max 8)
        if not _nan(row["pivot_s1"]):
            if price <= row["pivot_s1"]: gate_a_long += 4.0
            if price <= row["pivot_s2"]: gate_a_long += 4.0
            if price >= row["pivot_r1"]: gate_a_short += 4.0
            if price >= row["pivot_r2"]: gate_a_short += 4.0

        # ─────────────────────────────────────────────────────────
        # GATE B: Confirmation (Max 35 points)
        # Is momentum confirming a reversal?
        # ─────────────────────────────────────────────────────────
        gate_b_long, gate_b_short = 0.0, 0.0
        
        # 1. RSI (Max 15)
        if not _nan(row["rsi"]):
            if row["rsi"] <= self.profile.rsi_oversold_extreme: gate_b_long += 15.0
            elif row["rsi"] <= self.profile.rsi_oversold_mid: gate_b_long += 8.0
            if row["rsi"] >= self.profile.rsi_overbought_extreme: gate_b_short += 15.0
            elif row["rsi"] >= self.profile.rsi_overbought_mid: gate_b_short += 8.0

        # 2. Stochastic RSI (Max 10)
        if not _nan(row["stoch_rsi_k"], row["stoch_rsi_d"]):
            k, d = row["stoch_rsi_k"], row["stoch_rsi_d"]
            if k <= self.profile.stoch_oversold_extreme and k > d: gate_b_long += 10.0
            elif k <= self.profile.stoch_oversold_mid: gate_b_long += 5.0
            if k >= self.profile.stoch_overbought_extreme and k < d: gate_b_short += 10.0
            elif k >= self.profile.stoch_overbought_mid: gate_b_short += 5.0

        # 3. MACD (Max 10)
        if not _nan(row["macd_hist"]):
            if row["macd_hist"] > 0: gate_b_long += 10.0
            elif row["macd_hist"] > -atr * self.profile.macd_near_zero: gate_b_long += 5.0
            if row["macd_hist"] < 0: gate_b_short += 10.0
            elif row["macd_hist"] < atr * self.profile.macd_near_zero: gate_b_short += 5.0

        # ─────────────────────────────────────────────────────────
        # GATE C: Context (Max 25 points) - BONUS ONLY
        # Is the bigger picture helping?
        # ─────────────────────────────────────────────────────────
        gate_c_long, gate_c_short = 0.0, 0.0
        
        # 1. Macro Trend (Max 10)
        if not _nan(row["sma_200"]):
            if price > row["sma_200"]: gate_c_long += 10.0
            if price < row["sma_200"]: gate_c_short += 10.0

        # 2. VWAP (Max 7)
        if not _nan(row["vwap"]):
            if price > row["vwap"]: gate_c_long += 7.0
            if price < row["vwap"]: gate_c_short += 7.0

        # 3. Volume Confirmation (Max 8)
        if not _nan(row["volume_ratio"]):
            if row["volume_ratio"] >= self.profile.vol_ratio_surge:
                gate_c_long += 8.0
                gate_c_short += 8.0
            elif row["volume_ratio"] >= self.profile.vol_ratio_high:
                gate_c_long += 4.0
                gate_c_short += 4.0

        # ─────────────────────────────────────────────────────────
        # Final Processing
        # ─────────────────────────────────────────────────────────
        total_long  = gate_a_long + gate_b_long + gate_c_long
        total_short = gate_a_short + gate_b_short + gate_c_short

        # Determine dominant direction
        if total_long >= total_short:
            raw_score = total_long
            direction_bias = "LONG"
            mapped_probability = total_long  # Direct 0-100 scale for LONG
            breakdown = [
                {"name": "Gate A: Location",     "score": gate_a_long, "max": 40},
                {"name": "Gate B: Confirmation", "score": gate_b_long, "max": 35},
                {"name": "Gate C: Context",      "score": gate_c_long, "max": 25},
            ]
        else:
            raw_score = total_short
            direction_bias = "SHORT"
            mapped_probability = 100.0 - total_short # Invert for SHORT (e.g. 80 -> 20 <= 30)
            breakdown = [
                {"name": "Gate A: Location",     "score": gate_a_short, "max": 40},
                {"name": "Gate B: Confirmation", "score": gate_b_short, "max": 35},
                {"name": "Gate C: Context",      "score": gate_c_short, "max": 25},
            ]

        # Convert to signals
        if mapped_probability >= self.long_threshold and direction_bias == "LONG":
            direction = "LONG"
        elif mapped_probability <= self.short_threshold and direction_bias == "SHORT":
            direction = "SHORT"
        else:
            direction = "NO_TRADE"

        # Confidence
        if raw_score >= 82: confidence = "HIGH"
        elif raw_score >= 70: confidence = "MEDIUM"
        else: confidence = "LOW"

        # Reason text
        highest_gate = max(breakdown, key=lambda x: x["score"])
        reason = f"Dominant {direction_bias} setup driven by {highest_gate['name']} ({highest_gate['score']} pts)"

        sl, tp1, tp2, risk = _calculate_risk_levels(row, direction if direction != "NO_TRADE" else "LONG")

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

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        results = [self.score(row) for _, row in df.iterrows()]
        df["probability"] = [r.probability for r in results]
        df["direction"]   = [r.direction   for r in results]
        df["raw_score"]   = [r.raw_score   for r in results]
        return df

    def print_breakdown(self, signal: TradeSignal) -> None:
        print(f"\n  {'Gate / Signal':<45} {'Score':>7}  {'Max':>6}")
        print(f"  {'─' * 45} {'─' * 7}  {'─' * 6}")
        for b in signal.breakdown:
            print(f"  {b['name']:<45} {b['score']:>7.1f}  {b['max']:>6.1f}")
        print(f"  {'─' * 60}")
        print(f"  {'Total Probability':>45} {signal.probability:.1f}%")
