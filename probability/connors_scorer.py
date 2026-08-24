# ─────────────────────────────────────────────────────────────────
#  probability/connors_scorer.py
#  Larry Connors RSI(2) Mean Reversion Scorer for Cash Equities.
#
#  Rules:
#    1. Trend Filter: Close > SMA200 (Stock must be in secular uptrend)
#    2. Setup Trigger: 2-period RSI < 10.0 (Extreme short-term panic dip)
#    3. Target / Exit: Close > SMA5 (Fast snap-back to 5-day mean)
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from probability.signal_scorer import TradeSignal
from config.india_settings import INDIA_SIGNAL_THRESHOLD

IST = ZoneInfo("Asia/Kolkata")


class ConnorsScorer:
    """
    Larry Connors RSI(2) Ultra-Fast Pullback Scorer for Indian Cash Stocks.
    """

    def __init__(
        self,
        symbol:       str   = "RELIANCE.NS",
        threshold:    float = INDIA_SIGNAL_THRESHOLD,
        rsi_lookback: int   = 2,
        rsi_buy:      float = 12.0,   # Buy when RSI(2) < 12.0
        atr_sl_mult:  float = 1.5,
        atr_tp_mult:  float = 2.0,
        interval:     str   = "1d",
    ):
        self.symbol       = symbol
        self.threshold    = threshold
        self.rsi_lookback = rsi_lookback
        self.rsi_buy      = rsi_buy
        self.atr_sl_mult  = atr_sl_mult
        self.atr_tp_mult  = atr_tp_mult
        self.interval     = interval

    def score(self, row: pd.Series, df_slice: pd.DataFrame | None = None) -> TradeSignal:
        price = float(row["close"])
        if df_slice is None or len(df_slice) < 200:
            return self._no_trade(row, price, "Warmup < 200 bars for SMA200")

        # 1. Calculate 2-period RSI dynamically on df_slice
        closes = df_slice["close"].values
        diff   = np.diff(closes)
        gain   = np.where(diff > 0, diff, 0.0)
        loss   = np.where(diff < 0, -diff, 0.0)

        # Simple Wilder smoothing for 2 periods
        if len(gain) >= self.rsi_lookback:
            avg_gain = np.mean(gain[-self.rsi_lookback:])
            avg_loss = np.mean(loss[-self.rsi_lookback:])
            if avg_loss == 0:
                rsi2 = 100.0
            else:
                rs   = avg_gain / avg_loss
                rsi2 = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi2 = 50.0

        # 2. Trend Filter: Close > SMA200
        sma200 = _get(row, "sma_200")
        if _nan(sma200) or price <= sma200:
            return self._no_trade(row, price, f"Price={price:.1f} ≤ SMA200={sma200:.1f} (Not in secular uptrend)")

        # 3. Setup Trigger: RSI(2) < rsi_buy
        if rsi2 >= self.rsi_buy:
            return self._no_trade(row, price, f"RSI(2)={rsi2:.1f} ≥ {self.rsi_buy} (No panic dip)")

        # Strong signal fired!
        probability = 75.0 + min(20.0, (self.rsi_buy - rsi2) * 2.0)
        probability = min(95.0, probability)

        atr_val = _get(row, "atr")
        if _nan(atr_val) or atr_val <= 0:
            atr_val = price * 0.015

        sl_dist = atr_val * self.atr_sl_mult
        sl = round(price - sl_dist, 2)
        # In Connors RSI, TP is SMA5 or 2R
        sma5 = np.mean(closes[-5:]) if len(closes) >= 5 else price + sl_dist * self.atr_tp_mult
        tp   = round(max(price + sl_dist * 1.5, sma5), 2)

        reason = f"[CONNORS RSI(2) LONG] RSI(2)={rsi2:.1f} < {self.rsi_buy} in SMA200 Uptrend (Prob={probability:.1f}%)"

        return TradeSignal(
            symbol       = self.symbol,
            timestamp    = row.name if hasattr(row, "name") and row.name is not None else datetime.now(IST),
            direction    = "LONG",
            probability  = round(probability, 1),
            confidence   = "HIGH" if probability >= 80 else "MEDIUM",
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
