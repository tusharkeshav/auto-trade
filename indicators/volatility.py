# ─────────────────────────────────────────────────────────────────
#  indicators/volatility.py
#  Volatility indicators: Bollinger Bands, ATR
#
#  Volatility indicators measure how much price is swinging around.
#  High volatility = big moves (risky but opportunity-rich).
#  Low volatility  = tight range (often precedes a breakout).
# ─────────────────────────────────────────────────────────────────

import pandas as pd


# ── Bollinger Bands ───────────────────────────────────────────────

def add_bollinger_bands(
    df: pd.DataFrame,
    period: int    = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands — a volatility envelope around a moving average.

    Adds columns:
      - bb_middle    : SMA(20) — the mid-line
      - bb_upper     : middle + 2 × std   (resistance zone)
      - bb_lower     : middle − 2 × std   (support zone)
      - bb_width     : (upper − lower) / middle  → how wide the bands are
      - bb_pct       : where price sits within the bands (0=lower, 1=upper)

    Reading the signal:
      - Price touches lower band  → potential bounce up ✅
      - Price touches upper band  → potential pullback ⚠️
      - bb_width near multi-month low → "squeeze" → breakout likely coming
      - bb_pct < 0  → price below lower band (extreme oversold)
      - bb_pct > 1  → price above upper band (extreme overbought)
    """
    df = df.copy()

    rolling       = df["close"].rolling(window=period)
    df["bb_middle"] = rolling.mean()
    std             = rolling.std()

    df["bb_upper"]  = df["bb_middle"] + (std_dev * std)
    df["bb_lower"]  = df["bb_middle"] - (std_dev * std)

    df["bb_width"]  = ((df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]).round(4)
    df["bb_pct"]    = ((df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])).round(4)

    return df


# ── Average True Range ────────────────────────────────────────────

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    ATR — Average True Range.

    Measures the average price range per candle, accounting for gaps.
    This is the most important indicator for setting stop losses.

    Adds columns:
      - true_range : max of the three ranges below
      - atr        : smoothed average of true_range

    True Range = max of:
      1. High − Low              (normal candle range)
      2. |High − Prev Close|     (gap up candle)
      3. |Low  − Prev Close|     (gap down candle)

    Usage in our bot:
      stop_loss   = entry_price − (1.5 × ATR)
      take_profit = entry_price + (3.0 × ATR)   → 2:1 reward:risk
    """
    df = df.copy()

    prev_close = df["close"].shift(1)

    high_low        = df["high"] - df["low"]
    high_prev_close = (df["high"] - prev_close).abs()
    low_prev_close  = (df["low"]  - prev_close).abs()

    df["true_range"] = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df["atr"]        = df["true_range"].ewm(com=period - 1, min_periods=period).mean().round(2)
    
    # ATR as a percentage of price, to normalize across decades
    df["atr_pct"] = df["atr"] / df["close"] * 100
    
    # Rolling 1000-candle rank (0.0 to 1.0) to measure if we are in a high/low vol regime
    df["atr_percentile"] = df["atr_pct"].rolling(window=1000).rank(pct=True)

    return df
