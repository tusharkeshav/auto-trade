# ─────────────────────────────────────────────────────────────────
#  data/india/nse_options.py
#  Fetches NSE options chain and computes max pain.
#
#  NSE publishes options chain free at:
#    https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY
#    https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY
#
#  Max Pain: strike where total options writer loss is minimised.
#  At expiry, price gravitates toward max pain — especially last 2 days.
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests
from loguru import logger

_NSE_BASE     = "https://www.nseindia.com"
_CHAIN_URL    = _NSE_BASE + "/api/option-chain-indices?symbol={symbol}"
_HEADERS      = {
    "User-Agent":      "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/",
    "Connection":      "keep-alive",
}
_SESSION_URL  = _NSE_BASE + "/option-chain"   # warm-up page for cookie


@dataclass
class OptionsChainSnapshot:
    symbol:         str
    spot_price:     float
    expiry:         str          # nearest weekly expiry "DD-Mon-YYYY"
    max_pain:       float        # strike with min total writer loss
    pcr:            float        # put/call ratio by OI
    total_call_oi:  int
    total_put_oi:   int
    atm_iv:         float        # implied vol at ATM strike
    chain_df:       pd.DataFrame # full chain: strike, call_oi, put_oi, call_iv, put_iv

    @property
    def bias(self) -> str:
        """LONG if spot below max pain, SHORT if above, NEUTRAL within 0.5%."""
        diff_pct = (self.spot_price - self.max_pain) / self.max_pain * 100
        if diff_pct > 0.5:
            return "SHORT"   # spot above max pain → gravity pulls down
        if diff_pct < -0.5:
            return "LONG"    # spot below max pain → gravity pulls up
        return "NEUTRAL"

    @property
    def max_pain_distance_pct(self) -> float:
        return round((self.spot_price - self.max_pain) / self.max_pain * 100, 2)


class NSEOptionsClient:
    """
    Fetches NSE options chain with session cookie handling.
    NSE blocks requests without a valid browser session.
    """

    def __init__(self, timeout: int = 15):
        self._timeout  = timeout
        self._session  = requests.Session()
        self._session.headers.update(_HEADERS)
        self._warmed   = False

    def _warm_session(self) -> None:
        """Hit NSE homepage to get session cookies — required before API calls."""
        try:
            self._session.get(_SESSION_URL, timeout=self._timeout)
            time.sleep(1.5)
            self._warmed = True
            logger.debug("NSE session warmed")
        except Exception as e:
            logger.warning(f"NSE session warm-up failed: {e}")

    def get_chain(self, symbol: str = "NIFTY") -> dict:
        """Raw options chain JSON from NSE API."""
        if not self._warmed:
            self._warm_session()
        url  = _CHAIN_URL.format(symbol=symbol.upper())
        resp = self._session.get(url, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def get_snapshot(self, symbol: str = "NIFTY") -> OptionsChainSnapshot:
        """
        Parse NSE options chain into OptionsChainSnapshot.

        Args:
            symbol: "NIFTY" or "BANKNIFTY"
        """
        raw         = self.get_chain(symbol)
        records     = raw["records"]
        filtered    = raw["filtered"]
        spot_price  = float(records["underlyingValue"])
        expiry      = records["expiryDates"][0]   # nearest expiry

        rows = []
        for item in filtered.get("data", []):
            strike = item["strikePrice"]
            ce     = item.get("CE", {})
            pe     = item.get("PE", {})
            rows.append({
                "strike":   strike,
                "call_oi":  int(ce.get("openInterest", 0)),
                "put_oi":   int(pe.get("openInterest", 0)),
                "call_iv":  float(ce.get("impliedVolatility", 0)),
                "put_iv":   float(pe.get("impliedVolatility", 0)),
                "call_ltp": float(ce.get("lastPrice", 0)),
                "put_ltp":  float(pe.get("lastPrice", 0)),
            })

        chain_df        = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
        max_pain        = _compute_max_pain(chain_df)
        total_call_oi   = int(chain_df["call_oi"].sum())
        total_put_oi    = int(chain_df["put_oi"].sum())
        pcr             = round(total_put_oi / total_call_oi, 3) if total_call_oi else 0.0
        atm_iv          = _atm_iv(chain_df, spot_price)

        logger.info(
            f"[{symbol}] spot={spot_price:.0f}  max_pain={max_pain:.0f}  "
            f"PCR={pcr:.2f}  ATM_IV={atm_iv:.1f}%  expiry={expiry}"
        )

        return OptionsChainSnapshot(
            symbol        = symbol,
            spot_price    = spot_price,
            expiry        = expiry,
            max_pain      = max_pain,
            pcr           = pcr,
            total_call_oi = total_call_oi,
            total_put_oi  = total_put_oi,
            atm_iv        = atm_iv,
            chain_df      = chain_df,
        )


# ─────────────────────────────────────────────────────────────────
#  Max Pain
# ─────────────────────────────────────────────────────────────────

def _compute_max_pain(chain_df: pd.DataFrame) -> float:
    """
    Max Pain = strike where total loss for options writers is minimum.

    For each strike K (as hypothetical expiry price):
      - Call writers lose: sum over all strikes S < K of (K - S) * call_OI(S)
      - Put writers lose:  sum over all strikes S > K of (S - K) * put_OI(S)
      Total writer loss = call loss + put loss
    Max pain = K that minimises total writer loss.
    """
    strikes    = chain_df["strike"].values
    call_ois   = chain_df["call_oi"].values
    put_ois    = chain_df["put_oi"].values
    min_loss   = float("inf")
    max_pain   = strikes[len(strikes) // 2]

    for k in strikes:
        call_loss = sum(
            max(0, k - s) * oi
            for s, oi in zip(strikes, call_ois)
        )
        put_loss = sum(
            max(0, s - k) * oi
            for s, oi in zip(strikes, put_ois)
        )
        total = call_loss + put_loss
        if total < min_loss:
            min_loss = total
            max_pain = k

    return float(max_pain)


def _atm_iv(chain_df: pd.DataFrame, spot: float) -> float:
    """IV of the strike closest to spot (average of call + put IV)."""
    if chain_df.empty:
        return 0.0
    idx    = (chain_df["strike"] - spot).abs().idxmin()
    row    = chain_df.loc[idx]
    call_iv = row["call_iv"]
    put_iv  = row["put_iv"]
    iv_vals = [v for v in [call_iv, put_iv] if v > 0]
    return round(sum(iv_vals) / len(iv_vals), 2) if iv_vals else 0.0
