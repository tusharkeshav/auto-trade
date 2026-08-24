# ─────────────────────────────────────────────────────────────────
#  probability/india_momentum_scorer.py
#  Trend-following / momentum signal scorer for NIFTY/BANKNIFTY.
#
#  Used by RegimeSwitchScorer in low-VIX regime (VIX < 18).
#
#  Signals:
#    Gate A: EMA Trend Structure (EMA20 > EMA50 = uptrend)
#    Gate B: MACD Momentum Confirmation (MACD > MACD Signal)
#    Gate C: ADX Strength Filter (ADX > 20 = trending, not choppy)
#    Bonus : Breakout from 20-bar high/low adds probability
#
#  Research basis:
#    - Moskowitz, Ooi & Pedersen (2012): time-series momentum works
#      best in low-vol, trending environments (positive autocorrelation)
#    - Barroso & Santa-Clara (2015): momentum crashes in high-vol;
#      restricting momentum to VIX < 18 significantly improves Sharpe
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from probability.signal_scorer import TradeSignal
from config.india_settings import (
    INDIA_SIGNAL_THRESHOLD,
    INDIA_ATR_SL_MULT,
    INDIA_ATR_TP_MULT,
    NSE_MACRO_START_HOUR_IST, NSE_MACRO_START_MIN_IST,
    NSE_MACRO_END_HOUR_IST,   NSE_MACRO_END_MIN_IST,
)

IST = ZoneInfo("Asia/Kolkata")

# Momentum-specific thresholds
_ADX_TREND_MIN   = 18.0   # below this = consolidation, not trending enough for momentum
_ADX_STRONG      = 30.0   # above this = strong trend, higher probability
_EMA_FAST_COL    = "ema_12"
_EMA_SLOW_COL    = "ema_50"
_MACD_COL        = "macd"
_MACD_SIG_COL    = "macd_signal"
_BREAKOUT_PERIOD = 20     # bars for high/low lookback


class IndiaMomentumScorer:
    """
    Trend-following signal scorer for low-VIX regime in Indian equity indices.

    Generates LONG signals on uptrend breakouts/continuations and SHORT signals
    on downtrend breakdowns. Opposite philosophy to IndiaSignalScorer:
    buy strength, not weakness.
    """

    def __init__(
        self,
        symbol:       str   = "NIFTY50",
        threshold:    float = INDIA_SIGNAL_THRESHOLD,
        adx_min:      float = _ADX_TREND_MIN,
        atr_sl_mult:  float = INDIA_ATR_SL_MULT * 1.5,  # Momentum needs slightly wider SL (~1.875x)
        atr_tp_mult:  float = INDIA_ATR_TP_MULT * 1.5,  # Aim for higher R-multiple in trends (~2.25x)
        interval:     str   = "15m",
    ):
        self.symbol      = symbol
        self.threshold   = threshold
        self.adx_min     = adx_min
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.interval    = interval
        self._is_intraday = interval not in ("1d", "1wk", "1mo")

    def score(self, row: pd.Series, df_slice: pd.DataFrame | None = None) -> TradeSignal:
        """
        Score a single candle for momentum signal.

        Args:
            row      : current candle as pd.Series (with indicators)
            df_slice : optional DataFrame up to and including current row,
                       used for breakout computation (last N bars)
        """
        price = float(row["close"])
        adx   = _get(row, "adx")

        # ── Macro Shield 1: IST session window (intraday only) ─────────
        if self._is_intraday and hasattr(row, "name") and row.name is not None:
            ts_ist = _to_ist(row.name)
            if ts_ist is not None:
                session_start = ts_ist.replace(
                    hour=NSE_MACRO_START_HOUR_IST, minute=NSE_MACRO_START_MIN_IST,
                    second=0, microsecond=0
                )
                session_end = ts_ist.replace(
                    hour=NSE_MACRO_END_HOUR_IST, minute=NSE_MACRO_END_MIN_IST,
                    second=0, microsecond=0
                )
                if not (session_start <= ts_ist <= session_end):
                    return self._no_trade(
                        row, price,
                        f"IST={ts_ist.strftime('%H:%M')} outside session "
                        f"[{NSE_MACRO_START_HOUR_IST:02d}:{NSE_MACRO_START_MIN_IST:02d}"
                        f"–{NSE_MACRO_END_HOUR_IST:02d}:{NSE_MACRO_END_MIN_IST:02d} IST]"
                    )

        # ── Gate A: ADX confirms trend strength ───────────────────────
        if _nan(adx) or adx < self.adx_min:
            return self._no_trade(row, price, f"ADX={adx:.1f} < {self.adx_min} (choppy/no trend)")

        # ── Gate B: Trend direction via EMA crossover ─────────────────
        ema20 = _get(row, _EMA_FAST_COL)
        ema50 = _get(row, _EMA_SLOW_COL)
        if _nan(ema20) or _nan(ema50):
            return self._no_trade(row, price, "EMA20/50 unavailable")

        ema_up   = ema20 > ema50
        ema_down = ema20 < ema50
        if not ema_up and not ema_down:
            return self._no_trade(row, price, f"EMA20={ema20:.1f} EMA50={ema50:.1f} flat")

        # ── Gate C: MACD momentum confirmation ────────────────────────
        macd   = _get(row, _MACD_COL)
        macd_s = _get(row, _MACD_SIG_COL)
        if not _nan(macd) and not _nan(macd_s):
            macd_up   = macd > macd_s
            macd_down = macd < macd_s
        else:
            macd_up   = ema_up
            macd_down = ema_down

        if ema_up and macd_up:
            direction = "LONG"
        elif ema_down and macd_down:
            direction = "SHORT"
        elif ema_up:
            direction = "LONG"   # EMA dominant, MACD diverging
        else:
            direction = "SHORT"

        # ── Base probability from ADX strength ────────────────────────
        # ADX 20 = 55%, ADX 30+ = 75%, scale linearly
        adx_contrib = min(20.0, max(0.0, (adx - self.adx_min) / (_ADX_STRONG - self.adx_min) * 20.0))
        base_prob   = 55.0 + adx_contrib

        # MACD alignment bonus
        macd_aligned = (direction == "LONG" and macd_up) or (direction == "SHORT" and macd_down)
        macd_bonus   = 8.0 if macd_aligned else 0.0

        # Breakout bonus: price at N-bar high/low
        breakout_bonus = 0.0
        if df_slice is not None and len(df_slice) >= _BREAKOUT_PERIOD:
            window = df_slice.iloc[-_BREAKOUT_PERIOD:]
            n_high = float(window["high"].max())
            n_low  = float(window["low"].min())
            if direction == "LONG" and price >= n_high * 0.999:
                breakout_bonus = 7.0   # Price at/near N-bar high → momentum continuation
            elif direction == "SHORT" and price <= n_low * 1.001:
                breakout_bonus = 7.0

        probability = min(92.0, base_prob + macd_bonus + breakout_bonus)

        # Filter below threshold
        if probability < self.threshold:
            return self._no_trade(
                row, price,
                f"prob={probability:.1f}% < threshold={self.threshold} (dir={direction})"
            )

        confidence = "HIGH" if probability >= 75 else ("MEDIUM" if probability >= 65 else "LOW")

        # ── Compute SL/TP from ATR ────────────────────────────────────
        atr_val = _get(row, "atr")
        if _nan(atr_val) or atr_val <= 0:
            atr_val = price * 0.008

        sl_dist = atr_val * self.atr_sl_mult
        if direction == "LONG":
            sl = round(price - sl_dist, 2)
            tp = round(price + sl_dist * self.atr_tp_mult, 2)
        else:
            sl = round(price + sl_dist, 2)
            tp = round(price - sl_dist * self.atr_tp_mult, 2)

        reason = (
            f"Momentum {direction} | EMA20={'↑' if ema_up else '↓'} "
            f"ADX={adx:.1f} MACD={'✓' if macd_aligned else '⚠'} "
            f"prob={probability:.1f}% (base={base_prob:.0f} macd+={macd_bonus:.0f} brk+={breakout_bonus:.0f})"
        )

        return TradeSignal(
            symbol       = self.symbol,
            timestamp    = row.name if hasattr(row, "name") and row.name is not None else datetime.now(IST),
            direction    = direction,
            probability  = round(probability, 1),
            confidence   = confidence,
            raw_score    = round(probability, 1),
            entry_price  = round(price, 2),
            stop_loss    = sl,
            take_profit1 = tp,
            take_profit2 = tp,
            risk_amount  = round(sl_dist, 2),
            reason       = reason,
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


def _get(row: pd.Series, col: str) -> float:
    try:
        v = row[col]
        return float(v) if not (isinstance(v, float) and math.isnan(v)) else float("nan")
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _nan(v: float) -> bool:
    return math.isnan(v)


def _to_ist(ts) -> datetime | None:
    """Convert a pandas Timestamp (possibly UTC-aware) to IST datetime."""
    try:
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            return ts.to_pydatetime().astimezone(IST)
        from datetime import timezone
        return ts.to_pydatetime().replace(tzinfo=timezone.utc).astimezone(IST)
    except Exception:
        return None
