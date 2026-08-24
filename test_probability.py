# ─────────────────────────────────────────────────────────────────
#  test_probability.py
#  Scores the current BTC/USDT market and prints a full breakdown.
#
#  Usage:
#      source .venv/bin/activate && python test_probability.py
# ─────────────────────────────────────────────────────────────────

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.binance_client import BinanceClient
from indicators          import add_all_indicators
from probability         import SignalScorer


def section(title: str):
    print(f"\n{'═' * 65}")
    print(f"  {title}")
    print('═' * 65)


def main():
    client = BinanceClient()
    scorer = SignalScorer(symbol="BTCUSDT", long_threshold=70, short_threshold=30)

    # ── Fetch & compute ───────────────────────────────────────────
    section("FETCHING & COMPUTING — BTC/USDT 1h (300 candles)")
    df = client.get_ohlcv("BTCUSDT", "1h", 300)
    df = add_all_indicators(df)

    # Score the latest candle
    latest = df.iloc[-1]
    signal = scorer.score(latest)

    # ── Signal summary ────────────────────────────────────────────
    section(f"📊 TRADE SIGNAL — {signal.symbol}  @  {signal.timestamp}")

    direction_icon = {"LONG": "🟢", "SHORT": "🔴", "NO_TRADE": "⚪"}.get(signal.direction, "⚪")
    print(f"\n  Direction    : {direction_icon}  {signal.direction}")
    print(f"  Probability  : {signal.probability:.1f}%  ({signal.confidence} confidence)")
    print(f"  Raw Score    : {signal.raw_score:+.2f}  (range: −100 to +100)")
    print(f"  Top Signals  : {signal.reason}")

    if signal.is_tradeable():
        print(f"\n  ── Risk Levels ────────────────────────────────────────")
        print(f"  Entry Price  : ${signal.entry_price:>12,.2f}")
        print(f"  Stop Loss    : ${signal.stop_loss:>12,.2f}  ← 1 ATR below support (stop-hunt-safe)")
        print(f"  Risk Amount  : ${signal.risk_amount:>12,.2f}  per unit")
        print(f"  Take Profit1 : ${signal.take_profit1:>12,.2f}  ← book 50% here (1:1 R:R)")
        print(f"  Take Profit2 : ${signal.take_profit2:>12,.2f}  ← trail remaining 50% (1.5:1)")
        print(f"  After TP1    :  move stop to breakeven → zero-risk trade")
    else:
        print(f"\n  ⚠️  Probability {signal.probability:.1f}% is between 30–70%.")
        print(f"  Signals are not aligned enough. Skipping trade.")

    # ── Full breakdown ────────────────────────────────────────────
    section("🔬 SIGNAL BREAKDOWN (all 10 signals)")
    scorer.print_breakdown(signal)

    # ── Score last 10 candles ─────────────────────────────────────
    section("📈 LAST 10 CANDLES — Probability History")
    print(f"\n  {'Timestamp':<22} {'Close':>10}  {'Prob':>7}  {'Direction':<12}  Top Signal")
    print(f"  {'─' * 22} {'─' * 10}  {'─' * 7}  {'─' * 12}  ──────────")

    for ts, row in df.tail(10).iterrows():
        s = scorer.score(row)
        icon = {"LONG": "🟢", "SHORT": "🔴", "NO_TRADE": "⚪"}.get(s.direction, "⚪")
        top = s.breakdown and max(s.breakdown, key=lambda x: abs(x["contribution"]))
        top_name = top["signal"].split("|")[1].strip() if top else ""
        print(
            f"  {str(ts):<22}  "
            f"${row['close']:>10,.2f}  "
            f"{s.probability:>6.1f}%  "
            f"{icon} {s.direction:<10}  "
            f"{top_name}"
        )

    print()


if __name__ == "__main__":
    main()
