# ─────────────────────────────────────────────────────────────────
#  fetch_max_history.py
#  Fetches maximum free historical candles allowed by Yahoo/yfinance
#  and permanently stores them in `india_paper.sqlite -> india_ohlcv`.
# ─────────────────────────────────────────────────────────────────

import sqlite3
from loguru import logger
from data.india.nse_client import NSEClient
from india_paper_trade import SQLitePaperBroker, STOCK_ROUTING_MAP

def run_max_fetch():
    broker = SQLitePaperBroker()
    client = NSEClient()

    # Targets: ^NSEI + all 8 Mega-cap stocks
    symbols = ["^NSEI"] + list(STOCK_ROUTING_MAP.keys())

    logger.info(f"Starting maximum free historical fetch for {len(symbols)} Indian instruments...")

    for sym in symbols:
        logger.info(f"── Fetching maximum history for {sym} ──")

        # 1. Daily (1d) — get ~5 years (1250 bars)
        try:
            df_1d = client.get_ohlcv(sym, "1d", bars=1250)
            broker.save_ohlcv(sym, "1d", df_1d)
            logger.success(f"{sym} [1d]: Saved {len(df_1d) if df_1d is not None else 0} daily bars (~5 years).")
        except Exception as e:
            logger.error(f"{sym} [1d] error: {e}")

        # 2. Hourly (1h) — get ~2 years / 730 days max (3500 bars)
        try:
            df_1h = client.get_ohlcv(sym, "1h", bars=3500)
            broker.save_ohlcv(sym, "1h", df_1h)
            logger.success(f"{sym} [1h]: Saved {len(df_1h) if df_1h is not None else 0} hourly bars (~2 years).")
        except Exception as e:
            logger.error(f"{sym} [1h] error: {e}")

        # 3. 15-Minute (15m) — get ~60 days max (1500 bars)
        try:
            df_15m = client.get_ohlcv(sym, "15m", bars=1500)
            broker.save_ohlcv(sym, "15m", df_15m)
            logger.success(f"{sym} [15m]: Saved {len(df_15m) if df_15m is not None else 0} 15-min bars (~60 days max).")
        except Exception as e:
            logger.error(f"{sym} [15m] error: {e}")

    logger.info("Maximum history fetch complete! Checking SQLite database totals...")
    with sqlite3.connect("india_paper.sqlite") as conn:
        rows = conn.execute("SELECT symbol, interval, COUNT(*) FROM india_ohlcv GROUP BY symbol, interval").fetchall()
        for s, i, c in rows:
            print(f"  {s} [{i}]: {c:,} total candles stored.")

if __name__ == "__main__":
    run_max_fetch()
