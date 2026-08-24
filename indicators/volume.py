# ─────────────────────────────────────────────────────────────────
#  indicators/volume.py
#  Volume indicators: Volume MA, OBV, VWAP
#
#  Volume is the "fuel" behind price moves.
#  A price breakout with high volume is far more trustworthy
#  than the same move on low volume.
# ─────────────────────────────────────────────────────────────────

import pandas as pd


# ── Volume Moving Average ─────────────────────────────────────────

def add_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Volume Moving Average — rolling average of traded volume.

    Adds columns:
      - volume_ma   : average volume over N periods
      - volume_ratio: current volume / volume_ma  (relative volume)

    Reading the signal:
      - volume_ratio > 1.5 → unusually high volume (confirms the move)
      - volume_ratio < 0.5 → unusually low volume  (weak move, be cautious)
    """
    df = df.copy()

    df["volume_ma"]    = df["volume"].rolling(window=period).mean().round(2)
    df["volume_ratio"] = (df["volume"] / df["volume_ma"]).round(2)

    return df


# ── On-Balance Volume ─────────────────────────────────────────────

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    OBV — On-Balance Volume.

    Cumulative volume that flows in vs out. Tracks "smart money."

    Adds column: obv

    Logic:
      - If today's close > yesterday's close → buyers won → add volume
      - If today's close < yesterday's close → sellers won → subtract volume
      - If unchanged → add nothing

    Reading the signal:
      - OBV rising while price is flat     → accumulation, bullish divergence ✅
      - OBV falling while price is rising  → distribution, bearish divergence ⚠️
      - OBV confirming price direction     → trend is healthy
    """
    df = df.copy()

    direction = df["close"].diff().apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )

    df["obv"] = (direction * df["volume"]).cumsum().astype(int)

    return df


# ── VWAP ──────────────────────────────────────────────────────────

def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    VWAP — Volume Weighted Average Price.

    The average price at which ALL volume traded, weighted by how
    much was traded at each price. Resets at midnight UTC.

    Adds column: vwap

    Typical price = (high + low + close) / 3

    VWAP = cumulative(typical_price × volume) / cumulative(volume)

    Reading the signal:
      - Price above VWAP → market is bullish on the day ✅
      - Price below VWAP → market is bearish on the day ⚠️
      - VWAP acts as dynamic support/resistance intraday

    Note: Most meaningful on intraday timeframes (1m–1h).
          Less useful on daily/weekly charts.
    """
    df = df.copy()

    # Reset VWAP by date (it resets each trading day / UTC midnight)
    df["_date"]          = df.index.date
    df["_typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["_tpv"]           = df["_typical_price"] * df["volume"]

    df["vwap"] = (
        df.groupby("_date")["_tpv"].cumsum()
        / df.groupby("_date")["volume"].cumsum()
    ).round(2)

    # Clean up temporary columns
    df.drop(columns=["_date", "_typical_price", "_tpv"], inplace=True)

    return df
