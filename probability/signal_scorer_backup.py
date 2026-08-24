# ─────────────────────────────────────────────────────────────────
#  probability/signal_scorer.py
#
#  The Probability Engine.
#
#  Takes a single row of indicator values (from add_all_indicators)
#  and produces a trade signal with:
#    - A probability score (0–100)
#    - Direction: LONG | SHORT | NO_TRADE
#    - Full per-signal breakdown (transparency + debugging)
#    - Conservative risk levels: stop-hunt-safe SL + partial TP
#
#  Philosophy:
#    "Only trade when the odds are clearly in your favour."
#    Multiple independent signals must agree before we act.
#    We book small, safe profits and avoid getting hunted.
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


# ─────────────────────────────────────────────────────────────────
#  Trade Signal — the output of the engine
# ─────────────────────────────────────────────────────────────────

@dataclass
class TradeSignal:
    """
    Complete trade recommendation produced by the probability engine.

    Risk levels follow the user's conservative strategy:
      - Stop loss  : 1.0 × ATR *below* nearest support
                     (avoids stop-hunt zone where retail stops cluster)
      - Take profit 1: entry + 1.0 × risk  → book 50% here (safe profit)
      - Take profit 2: entry + 1.5 × risk  → let remaining 50% run
      - After TP1 hit: move stop to breakeven (zero-risk trade)
    """
    symbol:      str
    timestamp:   datetime

    direction:   str      # "LONG" | "SHORT" | "NO_TRADE"
    probability: float    # 0–100  (≥70 = trade, ≤30 = short, else skip)
    confidence:  str      # "HIGH" | "MEDIUM" | "LOW"
    raw_score:   float    # -100 to +100 (signed weighted sum)

    entry_price: float
    stop_loss:   float    # stop-hunt-safe: 1 ATR below nearest support
    take_profit1: float   # conservative: 1:1 R:R  (book 50% here)
    take_profit2: float   # extended:     1.5:1 R:R (trail rest)
    risk_amount:  float   # |entry - stop_loss|

    breakdown:   list[dict] = field(default_factory=list)
    reason:      str = ""   # plain-English summary of top signals

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


# ─────────────────────────────────────────────────────────────────
#  Individual signal scorers  (-1.0 → bearish, +1.0 → bullish)
#
#  Each function:
#    - Accepts a pandas Series (one row of the indicator DataFrame)
#    - Returns a float in [-1.0, +1.0]
#    - Returns 0.0 when data is missing (NaN warmup period)
# ─────────────────────────────────────────────────────────────────

def _nan(*values) -> bool:
    """True if any value is NaN or None."""
    return any(v is None or (isinstance(v, float) and math.isnan(v)) for v in values)


# ── Trend signals ─────────────────────────────────────────────────

def _score_sma200(row: pd.Series) -> float:
    """
    Long-term trend filter — graduated by distance.

    BUG FIX: Previously returned binary +1/-1. But being 0.1% below
    SMA200 is very different from being 5% below. A graduated score
    based on actual distance is more accurate.

    +1.0 when 3%+ above SMA200 (strong uptrend)
     0.0 when right at SMA200 (trend is neutral)
    -1.0 when 3%+ below SMA200 (strong downtrend)
    """
    if _nan(row["sma_200"]): return 0.0
    dist_pct = (row["close"] - row["sma_200"]) / row["sma_200"] * 100
    return max(-1.0, min(1.0, dist_pct / 3.0))


def _score_sma50(row: pd.Series) -> float:
    """
    Medium-term trend — graduated by distance.

    Same fix as SMA200: graduated instead of binary.
    Uses 2% distance for full score (SMA50 reacts faster).
    """
    if _nan(row["sma_50"]): return 0.0
    dist_pct = (row["close"] - row["sma_50"]) / row["sma_50"] * 100
    return max(-1.0, min(1.0, dist_pct / 2.0))


def _score_macd(row: pd.Series) -> float:
    """
    MACD histogram normalized by ATR.

    Dividing by ATR makes this price-agnostic — works the same
    whether BTC is at $30K or $100K.

    Histogram > 0 and growing = momentum building bullish.
    Histogram < 0 and falling = momentum building bearish.
    """
    if _nan(row["macd_hist"], row["atr"]) or row["atr"] == 0:
        return 0.0
    normalized = row["macd_hist"] / row["atr"]   # typically -2 to +2
    return max(-1.0, min(1.0, normalized))


# ── Momentum signals ──────────────────────────────────────────────

def _score_rsi(row: pd.Series) -> float:
    """
    RSI with graduated scoring — not just a binary threshold.

    RSI=30 = oversold → strong buy signal (+1.0)
    RSI=50 = neutral  → no signal (0.0)
    RSI=70 = overbought → strong sell signal (-1.0)
    """
    if _nan(row["rsi"]): return 0.0
    rsi = row["rsi"]

    if   rsi <= 25: return  1.0    # extreme oversold
    elif rsi <= 30: return  0.8
    elif rsi <= 40: return  0.4
    elif rsi <= 55: return  0.0    # neutral zone
    elif rsi <= 60: return -0.3
    elif rsi <= 70: return -0.6
    elif rsi <= 75: return -0.8
    else:           return -1.0    # extreme overbought


def _score_stoch_rsi(row: pd.Series) -> float:
    """
    Stochastic RSI — faster and more sensitive than plain RSI.

    Considers both the absolute level (oversold/overbought)
    and the %K vs %D crossover direction.
    """
    if _nan(row["stoch_rsi_k"], row["stoch_rsi_d"]): return 0.0
    k, d = row["stoch_rsi_k"], row["stoch_rsi_d"]

    # Base score from zone
    if   k <= 10: base =  1.0
    elif k <= 20: base =  0.7
    elif k <= 40: base =  0.2
    elif k <= 60: base =  0.0
    elif k <= 80: base = -0.4
    elif k <= 90: base = -0.7
    else:         base = -1.0

    # Bonus: K crossing above D = bullish momentum starting
    cross_bonus = 0.2 if k > d else -0.2

    return max(-1.0, min(1.0, base + cross_bonus))


# ── Volatility signals ────────────────────────────────────────────

def _score_bb_position(row: pd.Series) -> float:
    """
    Bollinger Band position (%B).

    %B = 0.0 → price at lower band (oversold)
    %B = 0.5 → price at midline (neutral)
    %B = 1.0 → price at upper band (overbought)
    %B > 1.0 → price above upper band (extended, risky)
    """
    if _nan(row["bb_pct"]): return 0.0
    bb = row["bb_pct"]

    if   bb <= 0.0: return  1.0    # at or below lower band
    elif bb <= 0.2: return  0.7
    elif bb <= 0.4: return  0.3
    elif bb <= 0.6: return  0.0    # neutral midzone
    elif bb <= 0.8: return -0.3
    elif bb <= 1.0: return -0.6
    else:           return -1.0    # above upper band: overextended


# ── Support / Resistance signals ──────────────────────────────────

def _score_sr_zone(row: pd.Series) -> float:
    """
    Support/Resistance proximity — the single highest-weight signal.

    BUG FIX: Previously used a fixed 0.3% zone which is too tight for
    fast timeframes (15m candle moves 0.3-0.5% per bar). Now uses ATR
    as the proximity measure, which automatically scales with both
    timeframe and current volatility.

    Within 0.5 ATR of support = strong buy (+1.0)
    Within 1.5 ATR of support = moderate buy (+0.3)
    Same logic mirrored for resistance.
    """
    if _nan(row["sr_support_dist_pct"], row["sr_resist_dist_pct"]):
        return 0.0

    price = row["close"]
    atr   = row["atr"] if not _nan(row["atr"]) else price * 0.004

    # Convert % distance back to absolute price distance
    sup_dist_abs = row["sr_support_dist_pct"] * price / 100
    res_dist_abs = row["sr_resist_dist_pct"]  * price / 100

    # Score based on ATR-relative proximity (scales with timeframe)
    bullish = 0.0
    if sup_dist_abs <= atr * 0.5:
        bullish = 1.0
    elif sup_dist_abs <= atr * 1.0:
        bullish = 0.6
    elif sup_dist_abs <= atr * 1.5:
        bullish = 0.3

    bearish = 0.0
    if res_dist_abs <= atr * 0.5:
        bearish = -1.0
    elif res_dist_abs <= atr * 1.0:
        bearish = -0.6
    elif res_dist_abs <= atr * 1.5:
        bearish = -0.3

    return max(-1.0, min(1.0, bullish + bearish))


def _score_pivot(row: pd.Series) -> float:
    """
    Pivot point position — is price above or below today's pivot?

    Above pivot (P) = bias is bullish for the session.
    Below pivot (P) = bias is bearish for the session.
    """
    if _nan(row["pivot_p"]): return 0.0
    price = row["close"]
    p  = row["pivot_p"]
    r1 = row["pivot_r1"]
    s1 = row["pivot_s1"]

    if   price > r1: return -0.5   # above R1: getting extended
    elif price > p:  return  0.5   # between P and R1: bullish bias
    elif price > s1: return -0.3   # between S1 and P: mild bearish
    else:            return  0.5   # below S1: oversold, potential bounce


# ── Volume signals ────────────────────────────────────────────────

def _score_volume(row: pd.Series) -> float:
    """
    Volume ratio vs 20-period average.

    High volume = conviction behind the move (confirms signal).
    Low volume  = weak move, unreliable signal (neutral score).

    BUG FIX: Previously capped at 0.4, meaning 4.8 out of 8 possible
    weighted points were unreachable. Now uses the full 0-0.8 range
    like every other non-directional signal. Still conservative:
    max is 0.8 (not 1.0) because volume confirms but doesn't create
    a directional bias by itself.
    """
    if _nan(row["volume_ratio"]): return 0.0
    ratio = row["volume_ratio"]

    if   ratio >= 3.0: return  0.8   # extreme volume: very strong confirmation
    elif ratio >= 2.0: return  0.6   # high volume: strong confirmation
    elif ratio >= 1.5: return  0.4   # above average: good confirmation
    elif ratio >= 1.0: return  0.1   # average: mild positive
    elif ratio >= 0.7: return  0.0   # slightly below average: neutral
    else:              return -0.3   # low volume: weak/suspicious move


def _score_vwap(row: pd.Series) -> float:
    """
    VWAP position — intraday bullish/bearish bias.

    Price above VWAP: buyers in control for the day.
    Price below VWAP: sellers in control for the day.
    """
    if _nan(row["vwap"]): return 0.0
    price = row["close"]
    vwap  = row["vwap"]
    diff_pct = (price - vwap) / vwap * 100

    if   diff_pct >  0.5: return  0.6
    elif diff_pct >  0.0: return  0.3
    elif diff_pct > -0.5: return -0.3
    else:                 return -0.6


# ─────────────────────────────────────────────────────────────────
#  Signal registry
#  Each entry: (scorer_function, weight, display_name)
#  Weights must sum to 100.
# ─────────────────────────────────────────────────────────────────

_SIGNALS: list[tuple] = [
    # ── Trend (28%) ──────────────────────────────────────────────
    (_score_sma200,      10, "Trend    | Price vs SMA 200 (macro)"),
    (_score_sma50,        8, "Trend    | Price vs SMA 50  (medium)"),
    (_score_macd,        10, "Trend    | MACD Histogram (ATR-normalised)"),

    # ── Momentum (22%) ───────────────────────────────────────────
    (_score_rsi,         12, "Momentum | RSI (14)"),
    (_score_stoch_rsi,   10, "Momentum | Stochastic RSI"),

    # ── Support / Resistance (25%) ────────────────────────────────
    (_score_sr_zone,     18, "S/R      | Swing Level Proximity"),
    (_score_pivot,        7, "S/R      | Daily Pivot Position"),

    # ── Volatility (10%) ─────────────────────────────────────────
    (_score_bb_position, 10, "Volatility| Bollinger Band Position"),

    # ── Volume (15%) ─────────────────────────────────────────────
    (_score_volume,       8, "Volume   | Volume Ratio vs MA"),
    (_score_vwap,         7, "Volume   | VWAP Position"),
]

# Verify weights sum to 100
assert sum(w for _, w, _ in _SIGNALS) == 100, "Signal weights must sum to 100"


# ─────────────────────────────────────────────────────────────────
#  Risk level calculator
# ─────────────────────────────────────────────────────────────────

def _calculate_risk_levels(
    row: pd.Series,
    direction: str,
) -> tuple[float, float, float, float]:
    """
    Returns (stop_loss, take_profit1, take_profit2, risk_amount).

    Stop-loss philosophy (anti-stop-hunt):
      - Place stop 1.0 × ATR BELOW nearest support (not at it)
      - This survives institutional liquidity sweeps that briefly
        dip below support before reversing

    Take-profit philosophy (conservative, "book small profits"):
      - TP1: 1.0 × risk → exit 50% of position here (guaranteed profit)
      - TP2: 1.5 × risk → let remaining 50% run to this level
      - After TP1 is hit → move stop to breakeven (zero-risk trade)
    """
    entry = row["close"]
    atr   = row["atr"] if not _nan(row["atr"]) else entry * 0.004

    sup = row["sr_support_price"]
    res = row["sr_resist_price"]

    if direction == "LONG":
        # Stop: 1 ATR below nearest support (stop-hunt-safe)
        base_support = sup if not _nan(sup) else (entry - atr)
        stop_loss    = base_support - (1.0 * atr)

        risk_amount  = entry - stop_loss
        take_profit1 = entry + (1.0 * risk_amount)   # 1:1 R:R → book 50%
        take_profit2 = entry + (1.5 * risk_amount)   # 1.5:1 → trail rest

    else:  # SHORT
        # Stop: 1 ATR above nearest resistance (stop-hunt-safe)
        base_resist  = res if not _nan(res) else (entry + atr)
        stop_loss    = base_resist + (1.0 * atr)

        risk_amount  = stop_loss - entry
        take_profit1 = entry - (1.0 * risk_amount)   # 1:1 R:R
        take_profit2 = entry - (1.5 * risk_amount)   # 1.5:1

    return round(stop_loss, 2), round(take_profit1, 2), round(take_profit2, 2), round(risk_amount, 2)


# ─────────────────────────────────────────────────────────────────
#  The Probability Engine
# ─────────────────────────────────────────────────────────────────

class SignalScorer:
    """
    Scores a single market snapshot and returns a TradeSignal.

    Usage:
        scorer = SignalScorer(symbol="BTCUSDT", long_threshold=70, short_threshold=30)
        signal = scorer.score(df.iloc[-1])

        if signal.is_tradeable():
            print(signal)
            scorer.print_breakdown(signal)
    """

    def __init__(
        self,
        symbol:           str   = "BTCUSDT",
        long_threshold:   float = 70.0,   # probability ≥ this → LONG
        short_threshold:  float = 30.0,   # probability ≤ this → SHORT
    ):
        self.symbol          = symbol
        self.long_threshold  = long_threshold
        self.short_threshold = short_threshold

    # ── Core scoring ──────────────────────────────────────────────

    def score(self, row: pd.Series) -> TradeSignal:
        """
        Score a single row of indicator data.

        Args:
            row: pd.Series — one row from a DataFrame with all indicators computed.
                 Use df.iloc[-1] for the most recent candle.

        Returns:
            TradeSignal with full breakdown and risk levels.
        """
        breakdown = []
        weighted_sum = 0.0

        for fn, weight, name in _SIGNALS:
            raw    = fn(row)
            contrib = raw * weight
            weighted_sum += contrib

            breakdown.append({
                "signal":       name,
                "raw_score":    round(raw, 3),     # -1 to +1
                "weight":       weight,
                "contribution": round(contrib, 3), # weight × raw
                "vote":         "🟢 BULL" if raw > 0.1 else ("🔴 BEAR" if raw < -0.1 else "⚪ NEUT"),
            })

        # weighted_sum is in range [-100, +100]
        # Map to probability [0, 100]
        probability = (weighted_sum + 100) / 2.0
        probability = round(max(0.0, min(100.0, probability)), 1)

        # Direction
        if probability >= self.long_threshold:
            direction = "LONG"
        elif probability <= self.short_threshold:
            direction = "SHORT"
        else:
            direction = "NO_TRADE"

        # Confidence
        if probability >= 82 or probability <= 18:
            confidence = "HIGH"
        elif probability >= 70 or probability <= 30:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # Risk levels
        sl, tp1, tp2, risk = _calculate_risk_levels(row, direction if direction != "NO_TRADE" else "LONG")

        # Build plain-English reason from top 3 contributors
        sorted_bd = sorted(breakdown, key=lambda x: abs(x["contribution"]), reverse=True)
        top_signals = [f"{b['signal'].split('|')[1].strip()} ({b['vote']})" for b in sorted_bd[:3]]
        reason = " · ".join(top_signals)

        return TradeSignal(
            symbol       = self.symbol,
            timestamp    = row.name if hasattr(row, "name") else datetime.utcnow(),
            direction    = direction,
            probability  = probability,
            confidence   = confidence,
            raw_score    = round(weighted_sum, 2),
            entry_price  = round(row["close"], 2),
            stop_loss    = sl,
            take_profit1 = tp1,
            take_profit2 = tp2,
            risk_amount  = risk,
            breakdown    = breakdown,
            reason       = reason,
        )

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score every row in a DataFrame.
        Useful for backtesting — adds a 'probability' and 'direction' column.

        Returns the original DataFrame with three new columns:
            probability, direction, raw_score
        """
        df = df.copy()
        results = [self.score(row) for _, row in df.iterrows()]

        df["probability"] = [r.probability for r in results]
        df["direction"]   = [r.direction   for r in results]
        df["raw_score"]   = [r.raw_score   for r in results]

        return df

    # ── Display helpers ───────────────────────────────────────────

    def print_breakdown(self, signal: TradeSignal) -> None:
        """Print a detailed per-signal breakdown table."""
        print(f"\n  {'Signal':<45} {'Score':>7}  {'Weight':>6}  {'Contrib':>8}  Vote")
        print(f"  {'─' * 45} {'─' * 7}  {'─' * 6}  {'─' * 8}  ────")
        for b in signal.breakdown:
            print(
                f"  {b['signal']:<45} "
                f"{b['raw_score']:>+7.3f}  "
                f"{b['weight']:>5}%  "
                f"{b['contribution']:>+8.3f}  "
                f"{b['vote']}"
            )
        print(f"  {'─' * 80}")
        print(f"  {'Raw weighted sum':>45} {signal.raw_score:>+7.2f}  → Probability: {signal.probability:.1f}%")
