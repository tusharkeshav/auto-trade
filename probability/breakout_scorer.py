# ─────────────────────────────────────────────────────────────────
#  probability/breakout_scorer.py
#  Donchian Channel (20-Day High) Breakout Scorer for Cash Equities.
#
#  Rules:
#    1. Macro Filter: Close > SMA200 (Long-term uptrend)
#    2. Setup Trigger: Close ≥ High of previous 20 days (Breakout)
#    3. Exit Target  : Close ≤ Low of previous 10 days OR 3.0× ATR trailing stop
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


class BreakoutScorer:
    """
    Donchian Channel Breakout Scorer for Indian Cash Stocks.
    """

    def __init__(
        self,
        symbol:       str   = "RELIANCE.NS",
        threshold:    float = INDIA_SIGNAL_THRESHOLD,
        entry_lookback: int = 20,   # 20-day High Breakout
        exit_lookback:  int = 10,   # 10-day Low Exit
        atr_sl_mult:  float = 1.5,
        atr_tp_mult:  float = 3.5,  # 1:3.5 Asymmetric R-multiple
        interval:     str   = "1d",
    ):
        self.symbol         = symbol
        self.threshold      = threshold
        self.entry_lookback = entry_lookback
        self.exit_lookback  = exit_lookback
        self.atr_sl_mult    = atr_sl_mult
        self.atr_tp_mult    = atr_tp_mult
        self.interval       = interval

    def score(self, row: pd.Series, df_slice: pd.DataFrame | None = None) -> TradeSignal:
        price = float(row["close"])
        if df_slice is None or len(df_slice) < 200:
            return self._no_trade(row, price, "Warmup < 200 bars for SMA200")

        # 1. Macro Filter: Close > SMA200
        sma200 = _get(row, "sma_200")
        if _nan(sma200) or price <= sma200:
            return self._no_trade(row, price, f"Price={price:.1f} ≤ SMA200={sma200:.1f} (Not in secular uptrend)")

        # 2. Check 20-Day High Breakout (excluding current bar)
        window = df_slice.iloc[-self.entry_lookback - 1 : -1]
        high_20 = float(window["high"].max())
        if price < high_20 * 0.999:
            return self._no_trade(row, price, f"Price={price:.1f} < 20-Day High={high_20:.1f}")

        # Breakout fired!
        adx = _get(row, "adx")
        adx_val = adx if not _nan(adx) else 20.0
        probability = 70.0 + min(25.0, (adx_val - 15.0) * 1.5)
        probability = min(95.0, probability)

        atr_val = _get(row, "atr")
        if _nan(atr_val) or atr_val <= 0:
            atr_val = price * 0.018

        sl_dist = atr_val * self.atr_sl_mult
        sl = round(price - sl_dist, 2)
        tp = round(price + sl_dist * self.atr_tp_mult, 2)

        reason = f"[DONCHIAN BREAKOUT LONG] Price={price:.1f} broke 20-Day High={high_20:.1f} (ADX={adx_val:.1f}, Prob={probability:.1f}%)"

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
