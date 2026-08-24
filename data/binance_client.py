# ─────────────────────────────────────────────────────────────────
#  data/binance_client.py
#  Wrapper around Binance public REST API.
#  No API key required — all endpoints here are publicly accessible.
# ─────────────────────────────────────────────────────────────────

import requests
import pandas as pd
from loguru import logger
from typing import Optional, Union

from config.settings import BINANCE_BASE_URL, DEFAULT_SYMBOL, DEFAULT_INTERVAL, DEFAULT_LIMIT


class BinanceClient:
    """
    Lightweight client for Binance public market data.

    All methods are read-only — no account or API key needed.
    Perfect for paper trading and backtesting.
    """

    BASE_URL = BINANCE_BASE_URL

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        logger.info("BinanceClient initialised — using public REST API (no key required)")

    # ─────────────────────────────────────────────
    #  Internal helper
    # ─────────────────────────────────────────────

    def _get(self, endpoint: str, params: Optional[dict] = None) -> Union[dict, list]:
        """Make a GET request and return parsed JSON. Raises on HTTP errors."""
        url = f"{self.BASE_URL}{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            logger.error("Connection failed — check your internet connection.")
            raise
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out after {self.timeout}s")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e} — Response: {response.text}")
            raise

    # ─────────────────────────────────────────────
    #  1. Server connectivity check
    # ─────────────────────────────────────────────

    def ping(self) -> bool:
        """Returns True if Binance API is reachable."""
        try:
            self._get("/api/v3/ping")
            logger.success("Binance API is reachable ✅")
            return True
        except Exception:
            logger.error("Binance API is NOT reachable ❌")
            return False

    def get_server_time(self) -> int:
        """Returns Binance server time as a Unix timestamp (milliseconds)."""
        data = self._get("/api/v3/time")
        server_time = data["serverTime"]
        logger.info(f"Binance server time: {pd.to_datetime(server_time, unit='ms')}")
        return server_time

    # ─────────────────────────────────────────────
    #  2. Live price
    # ─────────────────────────────────────────────

    def get_price(self, symbol: str = DEFAULT_SYMBOL) -> float:
        """
        Fetch the current market price for a symbol.

        Args:
            symbol: Trading pair, e.g. 'BTCUSDT'

        Returns:
            Current price as a float.
        """
        data = self._get("/api/v3/ticker/price", params={"symbol": symbol.upper()})
        price = float(data["price"])
        logger.info(f"{symbol.upper()} current price: ${price:,.2f}")
        return price

    def get_all_prices(self) -> pd.DataFrame:
        """Fetch current prices for ALL symbols on Binance. Returns a DataFrame."""
        data = self._get("/api/v3/ticker/price")
        df = pd.DataFrame(data)
        df["price"] = df["price"].astype(float)
        logger.info(f"Fetched prices for {len(df)} symbols")
        return df

    # ─────────────────────────────────────────────
    #  3. OHLCV (Candlestick) data
    # ─────────────────────────────────────────────

    def get_ohlcv(
        self,
        symbol: str = DEFAULT_SYMBOL,
        interval: str = DEFAULT_INTERVAL,
        limit: int = DEFAULT_LIMIT,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV (candlestick) data.

        Args:
            symbol:   Trading pair, e.g. 'BTCUSDT'
            interval: Candle size — '1m', '5m', '15m', '1h', '4h', '1d', etc.
            limit:    Number of candles to fetch (handles >1000 via pagination)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        all_klines = []
        end_time = None
        remaining = limit

        while remaining > 0:
            batch_limit = min(remaining, 1000)
            params = {
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": batch_limit,
            }
            if end_time:
                params["endTime"] = end_time

            raw = self._get("/api/v3/klines", params=params)
            if not raw:
                break
            
            all_klines = raw + all_klines
            remaining -= len(raw)
            # The first candle's open time minus 1 millisecond
            end_time = raw[0][0] - 1
            
            # If we didn't get a full batch, we've reached the start of the pair
            if len(raw) < batch_limit:
                break

        df = pd.DataFrame(all_klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])

        # Keep only the columns we need
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        df.set_index("timestamp", inplace=True)
        # Drop duplicates in case of overlap
        df = df[~df.index.duplicated(keep='first')].sort_index()
        
        logger.info(f"Fetched {len(df)} {interval} candles for {symbol.upper()}")
        return df

    # ─────────────────────────────────────────────
    #  4. 24-hour statistics
    # ─────────────────────────────────────────────

    def get_24h_stats(self, symbol: str = DEFAULT_SYMBOL) -> dict:
        """
        Fetch 24-hour rolling window statistics.

        Returns a dict with: price_change, price_change_pct, high, low,
                             volume, quote_volume, open, last_price
        """
        data = self._get("/api/v3/ticker/24hr", params={"symbol": symbol.upper()})

        stats = {
            "symbol":            data["symbol"],
            "last_price":        float(data["lastPrice"]),
            "open":              float(data["openPrice"]),
            "high":              float(data["highPrice"]),
            "low":               float(data["lowPrice"]),
            "price_change":      float(data["priceChange"]),
            "price_change_pct":  float(data["priceChangePercent"]),
            "volume":            float(data["volume"]),         # in base asset (BTC)
            "quote_volume":      float(data["quoteVolume"]),    # in USDT
            "trades":            int(data["count"]),
        }

        logger.info(
            f"{stats['symbol']} 24h — "
            f"Last: ${stats['last_price']:,.2f} | "
            f"Change: {stats['price_change_pct']:+.2f}% | "
            f"High: ${stats['high']:,.2f} | Low: ${stats['low']:,.2f}"
        )
        return stats

    # ─────────────────────────────────────────────
    #  5. Order book (market depth)
    # ─────────────────────────────────────────────

    def get_orderbook(self, symbol: str = DEFAULT_SYMBOL, depth: int = 5) -> dict:
        """
        Fetch top N bids and asks from the order book.

        Args:
            symbol: Trading pair
            depth:  Number of price levels to return (5, 10, 20, 50, 100, 500, 1000)

        Returns:
            dict with 'bids' and 'asks' as DataFrames (price, quantity)
        """
        data = self._get("/api/v3/depth", params={"symbol": symbol.upper(), "limit": depth})

        bids = pd.DataFrame(data["bids"], columns=["price", "qty"]).astype(float)
        asks = pd.DataFrame(data["asks"], columns=["price", "qty"]).astype(float)

        best_bid = bids["price"].iloc[0]
        best_ask = asks["price"].iloc[0]
        spread   = best_ask - best_bid
        spread_pct = (spread / best_ask) * 100

        logger.info(
            f"{symbol.upper()} Order Book — "
            f"Best Bid: ${best_bid:,.2f} | Best Ask: ${best_ask:,.2f} | "
            f"Spread: ${spread:.2f} ({spread_pct:.4f}%)"
        )

        return {"bids": bids, "asks": asks, "spread": spread, "spread_pct": spread_pct}
