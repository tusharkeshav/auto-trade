# ─────────────────────────────────────────────────────────────────
#  probability/expiry_pin_scorer.py
#  Options expiry pinning signal scorer.
#
#  Edge: NIFTY/BANKNIFTY tend to close near max pain on expiry day.
#  Max pain = strike where total options writer loss is minimised.
#  NSE weekly expiry = every Thursday. Trade window: Wed PM + Thu AM.
#
#  Signal logic:
#    - Only fire Wed 13:30+ IST or Thu 09:30-12:00 IST
#    - Compute distance from spot to max pain
#    - If spot far above max pain → SHORT (gravity pulls down)
#    - If spot far below max pain → LONG (gravity pulls up)
#    - Scale probability by: distance × PCR confirms × VIX multiplier
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from data.india.nse_options import OptionsChainSnapshot

IST = ZoneInfo("Asia/Kolkata")

# Expiry signal windows (IST)
_WED_ENTRY_HOUR    = 13   # Wed PM entry from 13:30
_WED_ENTRY_MIN     = 30
_THU_END_HOUR      = 12   # Thu AM exit by 12:00 (expiry usually 15:30 but pin effect weak after 12)
_THU_START_HOUR    = 9
_THU_START_MIN     = 30

# Min distance from max pain to trade (noise filter)
_MIN_DIST_PCT      = 0.3   # < 0.3% from max pain = NEUTRAL, no trade
_STRONG_DIST_PCT   = 0.8   # > 0.8% = strong signal

# PCR thresholds for confirmation
_BULLISH_PCR       = 0.8   # PCR < 0.8 = call-heavy = bearish options positioning
_BEARISH_PCR       = 1.2   # PCR > 1.2 = put-heavy = bullish options positioning


@dataclass
class ExpiryPinSignal:
    symbol:            str
    timestamp:         datetime
    direction:         str        # "LONG" | "SHORT" | "NO_TRADE"
    probability:       float      # 0-100
    confidence:        str        # "HIGH" | "MEDIUM" | "LOW"
    spot_price:        float
    max_pain:          float
    distance_pct:      float      # (spot - max_pain) / max_pain * 100
    pcr:               float
    atm_iv:            float
    reason:            str

    def is_tradeable(self) -> bool:
        return self.direction in ("LONG", "SHORT")


class ExpiryPinScorer:
    """
    Scores expiry pinning opportunity using live NSE options chain data.

    Parameters
    ----------
    min_dist_pct   : minimum distance from max pain to fire a signal
    vix_stress     : above this VIX, probability penalty applied
    """

    def __init__(
        self,
        symbol:       str   = "NIFTY",
        min_dist_pct: float = _MIN_DIST_PCT,
        vix_stress:   float = 25.0,
    ):
        self.symbol       = symbol
        self.min_dist_pct = min_dist_pct
        self.vix_stress   = vix_stress

    def score(
        self,
        snapshot:  OptionsChainSnapshot,
        now:       datetime | None = None,
        force:     bool = False,      # bypass time gate (for backtesting)
    ) -> ExpiryPinSignal:
        """
        Score a single options chain snapshot.

        Args:
            snapshot : OptionsChainSnapshot from NSEOptionsClient
            now      : current IST datetime (defaults to utcnow in IST)
            force    : if True, skip time-window gate (backtest mode)
        """
        now_ist = now or datetime.now(IST)

        # ── Time gate: only Wed PM or Thu AM ──────────────────────
        if not force and not _in_expiry_window(now_ist):
            return self._no_trade(snapshot, now_ist, f"Outside expiry window (IST={now_ist.strftime('%a %H:%M')})")

        dist_pct = snapshot.max_pain_distance_pct  # (spot - max_pain) / max_pain * 100
        spot     = snapshot.spot_price
        mp       = snapshot.max_pain
        pcr      = snapshot.pcr
        atm_iv   = snapshot.atm_iv

        # ── Noise filter ──────────────────────────────────────────
        if abs(dist_pct) < self.min_dist_pct:
            return self._no_trade(snapshot, now_ist,
                f"Spot {dist_pct:+.2f}% from max pain — inside noise band ±{self.min_dist_pct}%")

        # ── Direction: gravity toward max pain ────────────────────
        # spot > max_pain → price above, writers want it lower → SHORT
        # spot < max_pain → price below, writers want it higher → LONG
        direction = "SHORT" if dist_pct > 0 else "LONG"

        # ── Base probability from distance ────────────────────────
        # Scales from min_dist_pct (60%) to strong_dist_pct (85%) linearly
        dist_abs  = abs(dist_pct)
        base_prob = 60.0 + min(25.0, (dist_abs - self.min_dist_pct) / (_STRONG_DIST_PCT - self.min_dist_pct) * 25.0)

        # ── PCR confirmation bonus ─────────────────────────────────
        # SHORT signal + put-heavy PCR = confirming bearish options positioning
        # LONG signal  + call-heavy PCR = confirming bullish options positioning
        pcr_bonus = 0.0
        if direction == "SHORT" and pcr > _BEARISH_PCR:
            pcr_bonus = 5.0   # put OI heavy = writers positioned for downward pin
        elif direction == "LONG" and pcr < _BULLISH_PCR:
            pcr_bonus = 5.0   # call OI heavy = writers positioned for upward pin

        # ── VIX penalty ───────────────────────────────────────────
        # High VIX = erratic movement = expiry pin effect weaker
        vix_penalty = 0.0
        if atm_iv > self.vix_stress:
            vix_penalty = 10.0   # heavy VIX stress, reduce confidence
        elif atm_iv > self.vix_stress * 0.7:
            vix_penalty = 5.0

        probability = min(95.0, base_prob + pcr_bonus - vix_penalty)

        # ── Confidence ────────────────────────────────────────────
        if probability >= 80:
            confidence = "HIGH"
        elif probability >= 70:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        reason = (
            f"spot={spot:.0f} max_pain={mp:.0f} dist={dist_pct:+.2f}% "
            f"PCR={pcr:.2f} ATM_IV={atm_iv:.1f}% base={base_prob:.0f} "
            f"pcr_bonus={pcr_bonus:.0f} vix_pen={vix_penalty:.0f}"
        )

        return ExpiryPinSignal(
            symbol       = self.symbol,
            timestamp    = now_ist,
            direction    = direction,
            probability  = round(probability, 1),
            confidence   = confidence,
            spot_price   = spot,
            max_pain     = mp,
            distance_pct = dist_pct,
            pcr          = pcr,
            atm_iv       = atm_iv,
            reason       = reason,
        )

    def _no_trade(self, snapshot: OptionsChainSnapshot, now: datetime, reason: str) -> ExpiryPinSignal:
        return ExpiryPinSignal(
            symbol       = self.symbol,
            timestamp    = now,
            direction    = "NO_TRADE",
            probability  = 50.0,
            confidence   = "LOW",
            spot_price   = snapshot.spot_price,
            max_pain     = snapshot.max_pain,
            distance_pct = snapshot.max_pain_distance_pct,
            pcr          = snapshot.pcr,
            atm_iv       = snapshot.atm_iv,
            reason       = reason,
        )


def _in_expiry_window(now: datetime) -> bool:
    """
    True if now is within expiry trading window (IST):
      Wednesday from 13:30 onward
      Thursday  from 09:30 to 12:00
    """
    wd = now.weekday()   # 0=Monday … 6=Sunday
    h, m = now.hour, now.minute

    if wd == 2:   # Wednesday
        return (h, m) >= (_WED_ENTRY_HOUR, _WED_ENTRY_MIN)
    if wd == 3:   # Thursday
        if (h, m) < (_THU_START_HOUR, _THU_START_MIN):
            return False
        if h >= _THU_END_HOUR:
            return False
        return True
    return False
