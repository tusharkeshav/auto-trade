# ─────────────────────────────────────────────────────────────────
#  data/india/nse_client.py
#  Fetches OHLCV data for NSE/BSE instruments via yfinance.
#
#  Symbol conventions:
#    NSE equity  :  "RELIANCE.NS"
#    BSE equity  :  "RELIANCE.BO"
#    NIFTY 50    :  "^NSEI"
#    BANKNIFTY   :  "^NSEBANK"
#    India VIX   :  "^INDIAVIX"
#
#  Returns pandas DataFrame with columns:
#    open, high, low, close, volume
#  Index: UTC-aware DatetimeIndex
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from loguru import logger

IST = ZoneInfo("Asia/Kolkata")

# yfinance interval strings supported for intraday (max lookback in brackets)
VALID_INTERVALS = {
    "1m":  timedelta(days=7),
    "5m":  timedelta(days=60),
    "15m": timedelta(days=60),
    "30m": timedelta(days=60),
    "1h":  timedelta(days=730),
    "1d":  timedelta(days=3650),
    "1wk": timedelta(days=3650),
}

# Canonical NSE symbol map for key indices and commonly traded stocks
NSE_SYMBOLS = {
    # Indices
    "NIFTY50":    "^NSEI",
    "BANKNIFTY":  "^NSEBANK",
    "INDIAVIX":   "^INDIAVIX",
    "NIFTYIT":    "^CNXIT",
    "NIFTYMIDCAP":"^NSEMDCP50",

    # Nifty 50 heavy-weights (NSE suffix)
    "RELIANCE":   "RELIANCE.NS",
    "TCS":        "TCS.NS",
    "HDFCBANK":   "HDFCBANK.NS",
    "INFY":       "INFY.NS",
    "ICICIBANK":  "ICICIBANK.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "SBIN":       "SBIN.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "KOTAKBANK":  "KOTAKBANK.NS",
    "LT":         "LT.NS",
    "AXISBANK":   "AXISBANK.NS",
    "WIPRO":      "WIPRO.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "MARUTI":     "MARUTI.NS",
    "TITAN":      "TITAN.NS",
    "SUNPHARMA":  "SUNPHARMA.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS",
    "NESTLEIND":  "NESTLEIND.NS",
    "POWERGRID":  "POWERGRID.NS",
}

# NIFTY 50 constituent list (used for universe construction)
NIFTY50_LIST = [
    "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL", "BHARTIARTL",
    "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFOSYS",
    "ITC", "JSWSTEEL", "KOTAKBANK", "LT", "M&M",
    "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID",
    "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN", "SUNPHARMA",
    "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS", "TECHM",
    "TITAN", "TRENT", "ULTRACEMCO", "WIPRO", "ZOMATO",
]


def resolve_ticker(symbol: str) -> str:
    """
    Resolve a short symbol name to its yfinance ticker.
    Passes through if already looks like a full ticker (contains . or ^).
    """
    if "." in symbol or "^" in symbol:
        return symbol
    upper = symbol.upper()
    if upper in NSE_SYMBOLS:
        return NSE_SYMBOLS[upper]
    # Default: assume NSE equity
    return f"{upper}.NS"


class NSEClient:
    """
    Thin wrapper around yfinance for NSE/BSE OHLCV data.

    Usage:
        client = NSEClient()
        df = client.get_ohlcv("NIFTY50", "5m", bars=200)
        df = client.get_ohlcv("RELIANCE.NS", "1d", bars=500)
        vix = client.get_india_vix()
    """

    def __init__(self, retry_count: int = 3, retry_delay: float = 1.5):
        self._retry_count = retry_count
        self._retry_delay = retry_delay

    def get_ohlcv(
        self,
        symbol:   str,
        interval: str = "5m",
        bars:     int = 200,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for `symbol`.

        Args:
            symbol:   NSE short name (e.g. "NIFTY50", "RELIANCE") or full
                      yfinance ticker (e.g. "^NSEI", "RELIANCE.NS").
            interval: yfinance interval string ("1m","5m","15m","1h","1d").
            bars:     Approximate number of bars to return.

        Returns:
            DataFrame with columns [open, high, low, close, volume],
            DatetimeIndex in UTC.

        Raises:
            ValueError: Unknown interval or insufficient data.
            RuntimeError: yfinance fetch failed after retries.
        """
        if interval not in VALID_INTERVALS:
            raise ValueError(f"Interval '{interval}' not supported. Use: {list(VALID_INTERVALS)}")

        ticker  = resolve_ticker(symbol)
        max_lb  = VALID_INTERVALS[interval]

        # Estimate lookback period needed for requested bar count
        period_map = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "1d": 1440, "1wk": 10080
        }
        minutes_per_bar = period_map[interval]
        # NSE trades ~375 min/day, 5 days/week
        trading_minutes_per_day = 375
        days_needed = max(1, int((bars * minutes_per_bar) / trading_minutes_per_day) + 3)
        days_needed = min(days_needed, int(max_lb.days))

        end   = datetime.now(IST)
        start = end - timedelta(days=days_needed)

        df = self._fetch_with_retry(ticker, interval, start, end)
        df = self._normalize(df)

        if df.empty:
            raise RuntimeError(f"No data returned for {ticker} ({interval}) — check symbol and market hours")

        # Return the most recent `bars` rows
        return df.iloc[-bars:].copy()

    def get_price(self, symbol: str) -> float:
        """Latest close price for symbol."""
        df = self.get_ohlcv(symbol, interval="1m", bars=2)
        return float(df["close"].iloc[-1])

    def get_india_vix(self) -> float:
        """Current India VIX level."""
        df = self.get_ohlcv("INDIAVIX", interval="1d", bars=2)
        return float(df["close"].iloc[-1])

    def get_multiple(
        self,
        symbols:  list[str],
        interval: str = "5m",
        bars:     int  = 200,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for multiple symbols. Returns {symbol: DataFrame}."""
        result: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                result[sym] = self.get_ohlcv(sym, interval, bars)
            except Exception as e:
                logger.warning(f"Failed to fetch {sym}: {e}")
        return result

    # ── Internal helpers ──────────────────────────────────────────

    def _fetch_with_retry(
        self,
        ticker:   str,
        interval: str,
        start:    datetime,
        end:      datetime,
    ) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, self._retry_count + 1):
            try:
                tkr = yf.Ticker(ticker)
                df  = tkr.history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval=interval,
                    auto_adjust=True,
                    prepost=False,
                )
                return df
            except Exception as e:
                last_error = e
                logger.warning(f"yfinance retry {attempt}/{self._retry_count} for {ticker}: {e}")
                if attempt < self._retry_count:
                    time.sleep(self._retry_delay)
        raise RuntimeError(f"yfinance failed for {ticker} after {self._retry_count} retries") from last_error

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize yfinance output to lowercase OHLCV columns, UTC index."""
        if df.empty:
            return df

        # yfinance returns Title Case columns
        df = df.rename(columns={
            "Open":   "open",
            "High":   "high",
            "Low":    "low",
            "Close":  "close",
            "Volume": "volume",
        })

        # Keep only OHLCV
        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df   = df[cols].copy()

        # Ensure UTC index
        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Kolkata").tz_convert("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df = df.sort_index()
        df = df.dropna(subset=["open", "high", "low", "close"])
        return df
