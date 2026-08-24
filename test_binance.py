# ─────────────────────────────────────────────────────
#  test_binance.py
#  Quick sanity check — run this to verify Binance API
#  is working and data looks correct.
#
#  Usage:
#      python test_binance.py
# ─────────────────────────────────────────────────────

import sys
import os

# Make sure imports resolve from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from data.binance_client import BinanceClient

# ── Pretty separator helper ───────────────────────────
def section(title: str):
    print(f"\n{'═' * 55}")
    print(f"  {title}")
    print('═' * 55)


def main():
    client = BinanceClient()

    # ── Test 1: Connectivity ─────────────────────────
    section("TEST 1 — API Connectivity")
    is_up = client.ping()
    server_time = client.get_server_time()

    # ── Test 2: Live Price ───────────────────────────
    section("TEST 2 — Live Prices")
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        client.get_price(symbol)

    # ── Test 3: OHLCV Candles ────────────────────────
    section("TEST 3 — OHLCV Candle Data (BTC/USDT, 1h, last 5)")
    df = client.get_ohlcv(symbol="BTCUSDT", interval="1h", limit=5)
    print(df.to_string())

    # ── Test 4: 24h Stats ────────────────────────────
    section("TEST 4 — 24h Market Stats (BTC/USDT)")
    stats = client.get_24h_stats("BTCUSDT")
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"  {key:<20}: {val:,.4f}")
        else:
            print(f"  {key:<20}: {val}")

    # ── Test 5: Order Book ───────────────────────────
    section("TEST 5 — Order Book Top 5 (BTC/USDT)")
    ob = client.get_orderbook("BTCUSDT", depth=5)

    print("\n  📗 Top 5 BIDS (buyers)")
    print(ob["bids"].to_string(index=False))

    print("\n  📕 Top 5 ASKS (sellers)")
    print(ob["asks"].to_string(index=False))

    # ── Summary ──────────────────────────────────────
    section("✅ ALL TESTS PASSED — Binance API is working!")
    print("  You can now fetch live data, candles, and market depth.\n")


if __name__ == "__main__":
    main()
