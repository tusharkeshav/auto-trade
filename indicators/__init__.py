# indicators package — convenience re-exports
from .trend               import add_sma, add_ema, add_macd, add_adx
from .momentum            import add_rsi, add_stoch_rsi
from .volatility          import add_bollinger_bands, add_atr
from .volume              import add_volume_ma, add_obv, add_vwap
from .support_resistance  import add_support_resistance


def add_all_indicators(df):
    """
    Convenience function — computes every indicator in one call.
    Includes additional momentum/regime columns needed by both scorers.

    Usage:
        df = client.get_ohlcv("BTCUSDT", "1h", 300)
        df = add_all_indicators(df)

    Column groups added:
      Trend      : sma_20/50/200/800, ema_12/26/50, macd, macd_signal, macd_hist
      Momentum   : rsi, stoch_rsi_k, stoch_rsi_d
      Volatility : bb_upper/middle/lower/width/pct, true_range, atr
      Volume     : volume_ma, volume_ratio, obv, vwap
      S/R        : pivot_p/r1/r2/s1/s2,
                   sr_support_price, sr_resist_price,
                   sr_support_dist_pct, sr_resist_dist_pct,
                   sr_at_support, sr_at_resistance
      Momentum+  : di_plus, di_minus, atr_prev, macd_hist_prev,
                   high_20, low_20, bb_width_percentile,
                   bullish_candle_count, bearish_candle_count

    Requires at least ~200 candles for all indicators to be fully warmed up.
    """
    df = add_sma(df, periods=[20, 50, 200, 800])
    df = add_ema(df)
    df = add_macd(df)
    df = add_adx(df)           # now also exports di_plus, di_minus
    df = add_rsi(df)
    df = add_stoch_rsi(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)           # adds atr_percentile, true_range
    df = add_volume_ma(df)
    df = add_obv(df)
    df = add_vwap(df)
    df = add_support_resistance(df)   # pivot points + swing levels + proximity
    # ── Extra columns for momentum + regime detection ─────────
    df = _add_momentum_extras(df)
    return df


def _add_momentum_extras(df):
    """
    Adds supplementary columns used by MomentumScorer and regime detector.

    Columns:
      - atr_prev              : ATR shifted by 1 (for expansion comparison)
      - macd_hist_prev        : MACD histogram shifted by 1 (for momentum direction)
      - high_20               : highest high over last 20 candles
      - low_20                : lowest low over last 20 candles
      - bb_width_percentile   : rolling 1000-candle rank of BB width
      - bullish_candle_count  : consecutive candles where close > open
      - bearish_candle_count  : consecutive candles where close < open
    """
    df = df.copy()
    import numpy as np

    # ATR expansion
    df["atr_prev"] = df["atr"].shift(1)

    # MACD histogram direction
    df["macd_hist_prev"] = df["macd_hist"].shift(1)

    # 20-period range breakout
    df["high_20"] = df["high"].rolling(window=20).max()
    df["low_20"]  = df["low"].rolling(window=20).min()

    # BB width percentile (regime detector)
    bb_width_pct = df["bb_width"] / df["close"] * 100
    df["bb_width_percentile"] = bb_width_pct.rolling(window=1000).rank(pct=True)

    # Consecutive bullish/bearish candle count
    bullish = (df["close"] > df["open"]).astype(int)
    bearish = (df["close"] < df["open"]).astype(int)

    # Compute consecutive counts using groupby-cumsum reset
    df["bullish_candle_count"] = bullish.groupby((bullish == 0).cumsum()).cumsum()
    df["bearish_candle_count"] = bearish.groupby((bearish == 0).cumsum()).cumsum()

    return df
