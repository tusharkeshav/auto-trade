# ─────────────────────────────────────────────────────────────────
#  indicators/momentum.py
#  Momentum indicators: RSI, Stochastic RSI
#
#  Momentum indicators measure the speed and strength of price moves.
#  They tell you if a move is "too far, too fast" (overbought/oversold).
# ─────────────────────────────────────────────────────────────────

import pandas as pd


# ── RSI ───────────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    RSI — Relative Strength Index (0 to 100).

    Adds column: rsi

    Reading the signal:
      - RSI > 70  → overbought  (price may reverse down soon) ⚠️
      - RSI < 30  → oversold    (price may bounce up soon)    ✅
      - RSI = 50  → neutral

    Math:
      avg_gain / avg_loss over the last N periods → RS
      RSI = 100 − (100 / (1 + RS))
    """
    df = df.copy()

    delta = df["close"].diff()

    gain = delta.clip(lower=0)           # only keep positive moves
    loss = delta.clip(upper=0).abs()     # only keep negative moves (as positive)

    # Use Wilder's smoothing (equivalent to EWM with alpha = 1/period)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs          = avg_gain / avg_loss
    df["rsi"]   = (100 - (100 / (1 + rs))).round(2)

    return df


# ── Stochastic RSI ────────────────────────────────────────────────

def add_stoch_rsi(
    df: pd.DataFrame,
    rsi_period: int  = 14,
    stoch_period: int = 14,
    k_smooth: int    = 3,
    d_smooth: int    = 3,
) -> pd.DataFrame:
    """
    Stochastic RSI — applies Stochastic formula on top of RSI.

    Adds columns:
      - stoch_rsi_k : fast line (0–100)
      - stoch_rsi_d : slow signal line, SMA of K (0–100)

    More sensitive than plain RSI — gives earlier signals.

    Reading the signal:
      - K < 20          → oversold zone ✅
      - K > 80          → overbought zone ⚠️
      - K crossing above D → bullish signal
      - K crossing below D → bearish signal
    """
    df = df.copy()

    # First compute RSI (we need it as a series)
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = delta.clip(upper=0).abs()
    avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    rs       = avg_gain / avg_loss
    rsi      = 100 - (100 / (1 + rs))

    # Apply Stochastic formula on the RSI values
    rsi_min = rsi.rolling(window=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period).max()

    raw_k = 100 * (rsi - rsi_min) / (rsi_max - rsi_min)

    df["stoch_rsi_k"] = raw_k.rolling(window=k_smooth).mean().round(2)
    df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(window=d_smooth).mean().round(2)

    return df
