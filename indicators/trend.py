# ─────────────────────────────────────────────────────────────────
#  indicators/trend.py
#  Trend indicators: SMA, EMA, MACD
#
#  All functions accept a DataFrame with at minimum a 'close' column
#  and return the same DataFrame with new indicator columns appended.
#  Nothing is modified in place — a copy is always returned.
# ─────────────────────────────────────────────────────────────────

import pandas as pd


# ── Simple Moving Average ─────────────────────────────────────────

def add_sma(df: pd.DataFrame, periods: list[int] = [20, 50, 200]) -> pd.DataFrame:
    """
    Simple Moving Average — unweighted mean of the last N closes.

    Adds columns: sma_20, sma_50, sma_200  (or whatever periods you pass)

    Reading the signal:
      - Price above SMA → uptrend
      - Price below SMA → downtrend
      - SMA_20 crossing above SMA_50 → "golden cross" (bullish)
      - SMA_20 crossing below SMA_50 → "death cross" (bearish)
    """
    df = df.copy()
    for period in periods:
        df[f"sma_{period}"] = df["close"].rolling(window=period).mean()
    return df


# ── Exponential Moving Average ────────────────────────────────────

def add_ema(df: pd.DataFrame, periods: list[int] = [12, 26, 50]) -> pd.DataFrame:
    """
    Exponential Moving Average — gives more weight to recent prices.

    Adds columns: ema_12, ema_26, ema_50

    Compared to SMA, EMA reacts faster to price changes.
    Used internally by MACD (ema_12 and ema_26).
    """
    df = df.copy()
    for period in periods:
        df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
    return df


# ── MACD ──────────────────────────────────────────────────────────

def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD — Moving Average Convergence Divergence.

    Adds columns:
      - macd        : EMA(fast) − EMA(slow)
      - macd_signal : EMA(signal) of macd line
      - macd_hist   : macd − macd_signal  (the "histogram")

    Reading the signal:
      - macd_hist > 0 and rising  → bullish momentum building
      - macd_hist < 0 and falling → bearish momentum building
      - macd crossing above signal line → bullish crossover ✅
      - macd crossing below signal line → bearish crossover ❌
    """
    df = df.copy()

    ema_fast   = df["close"].ewm(span=fast,   adjust=False).mean()
    ema_slow   = df["close"].ewm(span=slow,   adjust=False).mean()

    df["macd"]        = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    return df


# ── Average Directional Index (ADX) ───────────────────────────────

def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average Directional Index (ADX).
    Measures the strength of a trend, regardless of direction.
    
    Adds column: adx
    
    Reading the signal:
      - ADX < 20 : Weak or non-existent trend (good for mean reversion).
      - ADX > 25 : Strong trend forming.
      - ADX > 30 : Very strong trend (bad for counter-trend mean reversion).
    """
    import numpy as np
    df = df.copy()
    
    high = df['high']
    low = df['low']
    close = df['close']

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift()
    down_move = low.shift() - low

    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Smoothed
    tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
    pos_dm_smooth = pd.Series(pos_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    neg_dm_smooth = pd.Series(neg_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()

    # Directional Indicators
    di_plus = 100 * (pos_dm_smooth / tr_smooth)
    di_minus = 100 * (neg_dm_smooth / tr_smooth)

    # ADX
    dx = 100 * (np.abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, 1))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    df['adx'] = adx
    # Export DMI components for momentum strategy
    df['di_plus'] = di_plus
    df['di_minus'] = di_minus
    return df
