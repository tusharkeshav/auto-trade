# ─────────────────────────────────────────────────────────────────
#  test_indicators.py
#  Fetch live BTC/USDT data and verify all indicators compute
#  correctly. Prints a clean summary of current market state.
#
#  Usage:
#      python test_indicators.py
# ─────────────────────────────────────────────────────────────────

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.binance_client import BinanceClient
from indicators import add_all_indicators


# ── Helpers ───────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print('═' * 60)

def signal_label(value: float, low: float, high: float) -> str:
    """Returns a colored label based on where value sits in [low, high]."""
    if value <= low:  return "🟢 OVERSOLD  (potential BUY zone)"
    if value >= high: return "🔴 OVERBOUGHT (potential SELL zone)"
    return              "⚪ NEUTRAL"


# ── Main ──────────────────────────────────────────────────────────

def main():
    client = BinanceClient()

    # Fetch 300 candles — enough to warm up all indicators (200 EMA needs ~200)
    section("FETCHING DATA — BTC/USDT 1h (last 300 candles)")
    df = client.get_ohlcv(symbol="BTCUSDT", interval="1h", limit=300)
    print(f"  Candles fetched : {len(df)}")
    print(f"  From            : {df.index[0]}")
    print(f"  To              : {df.index[-1]}")

    # Compute all indicators in one call
    section("COMPUTING INDICATORS")
    df = add_all_indicators(df)
    print(f"  Total columns   : {len(df.columns)}")
    print(f"  Columns         : {', '.join(df.columns.tolist())}")

    # ── Snapshot: last 3 candles ──────────────────────────────────
    section("LAST 3 CANDLES WITH INDICATORS")
    display_cols = ["close", "rsi", "macd_hist", "bb_pct", "atr", "volume_ratio", "vwap"]
    print(df[display_cols].tail(3).to_string())

    # ── Current Market State ──────────────────────────────────────
    latest = df.iloc[-1]
    section("📊 CURRENT MARKET STATE — BTC/USDT")

    price = latest["close"]
    print(f"\n  Price           : ${price:>12,.2f}")
    print(f"  VWAP            : ${latest['vwap']:>12,.2f}  {'↑ above VWAP (bullish)' if price > latest['vwap'] else '↓ below VWAP (bearish)'}")

    print(f"\n  ── Trend ────────────────────────────────────────────")
    print(f"  SMA 20          : ${latest['sma_20']:>12,.2f}  {'✅ price above' if price > latest['sma_20'] else '❌ price below'}")
    print(f"  SMA 50          : ${latest['sma_50']:>12,.2f}  {'✅ price above' if price > latest['sma_50'] else '❌ price below'}")
    print(f"  SMA 200         : ${latest['sma_200']:>12,.2f}  {'✅ price above' if price > latest['sma_200'] else '❌ price below'}")
    print(f"  EMA 12          : ${latest['ema_12']:>12,.2f}")
    print(f"  EMA 26          : ${latest['ema_26']:>12,.2f}")
    macd_bias = "🟢 bullish" if latest["macd_hist"] > 0 else "🔴 bearish"
    print(f"  MACD Histogram  : {latest['macd_hist']:>12.4f}  {macd_bias}")

    print(f"\n  ── Momentum ─────────────────────────────────────────")
    print(f"  RSI (14)        : {latest['rsi']:>12.2f}  {signal_label(latest['rsi'], 30, 70)}")
    print(f"  Stoch RSI %K    : {latest['stoch_rsi_k']:>12.2f}  {signal_label(latest['stoch_rsi_k'], 20, 80)}")
    print(f"  Stoch RSI %D    : {latest['stoch_rsi_d']:>12.2f}")

    print(f"\n  ── Volatility ───────────────────────────────────────")
    print(f"  BB Upper        : ${latest['bb_upper']:>12,.2f}")
    print(f"  BB Middle       : ${latest['bb_middle']:>12,.2f}")
    print(f"  BB Lower        : ${latest['bb_lower']:>12,.2f}")
    print(f"  BB %            : {latest['bb_pct']:>12.4f}  (0=lower band, 1=upper band)")
    print(f"  BB Width        : {latest['bb_width']:>12.4f}  (lower = tighter = possible squeeze)")
    print(f"  ATR (14)        : ${latest['atr']:>12.2f}")

    sl_long  = price - (1.5 * latest["atr"])
    tp_long  = price + (3.0 * latest["atr"])
    print(f"\n  ATR-based levels (if going LONG now):")
    print(f"    Stop Loss     : ${sl_long:>12,.2f}  (−1.5 × ATR)")
    print(f"    Take Profit   : ${tp_long:>12,.2f}  (+3.0 × ATR)  → 2:1 R:R")

    print(f"\n  ── Volume ───────────────────────────────────────────")
    vol_label = "🔥 high volume" if latest["volume_ratio"] > 1.5 else ("💤 low volume" if latest["volume_ratio"] < 0.7 else "normal")
    print(f"  Volume Ratio    : {latest['volume_ratio']:>12.2f}  {vol_label}")
    print(f"  OBV             : {latest['obv']:>12,.0f}")

    print(f"\n  ── Support & Resistance ─────────────────────────────")
    print(f"  Pivot (P)       : ${latest['pivot_p']:>12,.2f}")
    print(f"  Resistance 1    : ${latest['pivot_r1']:>12,.2f}")
    print(f"  Resistance 2    : ${latest['pivot_r2']:>12,.2f}")
    print(f"  Support 1       : ${latest['pivot_s1']:>12,.2f}")
    print(f"  Support 2       : ${latest['pivot_s2']:>12,.2f}")
    print(f"  Nearest Support : ${latest['sr_support_price']:>12,.2f}  ({latest['sr_support_dist_pct']:.3f}% below price)")
    print(f"  Nearest Resist  : ${latest['sr_resist_price']:>12,.2f}  ({latest['sr_resist_dist_pct']:.3f}% above price)")
    print(f"  At Support Zone : {'🟢 YES — potential bounce' if latest['sr_at_support'] else 'No'}")
    print(f"  At Resist Zone  : {'🔴 YES — potential rejection' if latest['sr_at_resistance'] else 'No'}")

    # Stop-hunt-safe stop loss: 1.0×ATR BELOW the nearest support (not at it)
    # This puts our stop BELOW where institutions sweep retail stops
    safe_sl = latest["sr_support_price"] - (1.0 * latest["atr"])
    print(f"\n  Stop-Hunt-Safe SL : ${safe_sl:>12,.2f}  (1×ATR below nearest support)")

    # ── Data integrity check ──────────────────────────────────────
    section("✅ INDICATOR HEALTH CHECK")
    nan_counts = df[["rsi", "macd", "bb_upper", "atr", "obv", "vwap"]].isna().sum()
    all_ok = True
    for col, count in nan_counts.items():
        status = "✅" if count < 50 else "⚠️ "   # some NaNs at start are expected (warmup)
        print(f"  {col:<15}: {count:>3} NaN rows  {status}")
        if count > 200:
            all_ok = False

    if all_ok:
        print("\n  All indicators computed successfully! ✅")
    print()


if __name__ == "__main__":
    main()
