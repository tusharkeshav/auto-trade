# ─────────────────────────────────────────────────────────────────
#  data/binance_futures.py
#
#  Binance USDⓈ-M Futures API — public endpoints only.
#  No API key required for market data, funding rate, and OI.
#
#  Funding rate is the key signal: extreme values (±0.05%+)
#  indicate retail over-leverage → mean reversion opportunity.
# ─────────────────────────────────────────────────────────────────

import requests
import pandas as pd
from datetime import datetime
from loguru import logger
from typing import Optional

from config.settings import BINANCE_FUTURES_URL


class BinanceFuturesClient:
    """
    Public market data client for Binance USDⓈ-M futures.

    Covers:
      - Funding rate history
      - Current premium index (mark price, funding rate, next funding time)
      - Open interest
    """

    BASE_URL = BINANCE_FUTURES_URL

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        logger.info("BinanceFuturesClient initialised — public futures API")

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict | list:
        url = f"{self.BASE_URL}{endpoint}"
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Futures API error: {e}")
            raise

    # ─────────────────────────────────────────────────────────
    #  Funding Rate
    # ─────────────────────────────────────────────────────────

    def get_current_funding_info(self, symbol: str = "BTCUSDT") -> dict:
        """
        Current premium index for a symbol.

        Returns:
            {
                "symbol": "BTCUSDT",
                "markPrice": "65432.1",
                "lastFundingRate": "0.0001",
                "nextFundingTime": 1719878400000,
                "interestRate": "0.00010000"
            }
        """
        return self._get("/fapi/v1/premiumIndex", {"symbol": symbol})

    def get_funding_rate_history(
        self,
        symbol: str = "BTCUSDT",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Historical funding rate data.

        Args:
            symbol: Trading pair
            limit: Max 1000 per request
            start_time: milliseconds timestamp
            end_time: milliseconds timestamp

        Returns:
            DataFrame with columns: symbol, fundingTime, fundingRate, markPrice
            fundingRate is a decimal (0.0001 = 0.01%)
        """
        params = {"symbol": symbol, "limit": min(limit, 1000)}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        data = self._get("/fapi/v1/fundingRate", params)

        df = pd.DataFrame(data)
        if df.empty:
            return df

        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce").fillna(0.0)
        df["markPrice"] = pd.to_numeric(df["markPrice"], errors="coerce").fillna(0.0)
        return df

    def get_funding_rate_history_bulk(
        self,
        symbol: str = "BTCUSDT",
        total_candles: int = 10000,
    ) -> pd.DataFrame:
        """
        Fetch full funding rate history by paginating 1000 at a time.
        Binance keeps ~300 days of funding data (every 8h = ~2700 entries).
        """
        all_dfs = []
        end_time = None

        while len(all_dfs) * 1000 < total_candles:
            df = self.get_funding_rate_history(
                symbol=symbol,
                limit=1000,
                end_time=end_time,
            )
            if df.empty:
                break
            all_dfs.append(df)
            end_time = int(df["fundingTime"].iloc[0].timestamp() * 1000)

        if not all_dfs:
            return pd.DataFrame()

        result = pd.concat(all_dfs).drop_duplicates(subset="fundingTime").reset_index(drop=True)
        result = result.sort_values("fundingTime").reset_index(drop=True)
        logger.info(f"Fetched {len(result)} funding records for {symbol}")
        return result

    # ─────────────────────────────────────────────────────────
    #  Open Interest
    # ─────────────────────────────────────────────────────────

    def get_open_interest(self, symbol: str = "BTCUSDT") -> float:
        """
        Current total open interest in USDT.
        """
        data = self._get("/fapi/v1/openInterest", {"symbol": symbol})
        return float(data.get("openInterest", 0))

    def get_open_interest_history(
        self,
        symbol: str = "BTCUSDT",
        period: str = "15m",
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Historical open interest.

        Period options: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
        """
        data = self._get("/futures/data/openInterestHist", {
            "symbol": symbol,
            "period": period,
            "limit": min(limit, 500),
        })
        df = pd.DataFrame(data)
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
        df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
        return df

    # ─────────────────────────────────────────────────────────
    #  Taker Buy/Sell Volume
    # ─────────────────────────────────────────────────────────

    def get_taker_volume(self, symbol: str = "BTCUSDT", period: str = "15m", limit: int = 500) -> pd.DataFrame:
        """
        Taker buy/sell volume ratio.

        buySellRatio = takerBuyVol / takerSellVol
        > 1.0 → aggressive buying
        < 1.0 → aggressive selling
        """
        data = self._get("/futures/data/takerlongshortRatio", {
            "symbol": symbol,
            "period": period,
            "limit": min(limit, 500),
        })
        df = pd.DataFrame(data)
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["buySellRatio"] = df["buySellRatio"].astype(float)
        df["buyVol"] = df["takerBuyVol"].astype(float)
        df["sellVol"] = df["takerSellVol"].astype(float)
        return df


if __name__ == "__main__":
    # Quick test
    fc = BinanceFuturesClient()

    info = fc.get_current_funding_info()
    rate_pct = float(info["lastFundingRate"]) * 100
    print(f"Current BTC funding rate: {rate_pct:.4f}%")
    print(f"Next funding at: {datetime.fromtimestamp(info['nextFundingTime']/1000)}")

    hist = fc.get_funding_rate_history(limit=10)
    print(f"\nRecent funding rates:")
    print(hist[["fundingTime", "fundingRate"]].to_string(index=False))

    oi = fc.get_open_interest()
    print(f"\nOpen Interest: {oi:,.0f} USDT")
