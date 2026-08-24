# ─────────────────────────────────────────────────────────────────
#  engine/india_costs.py
#  Indian equity/index transaction cost calculator.
#
#  Components (all mandatory, regulatory):
#    STT      — Securities Transaction Tax (on sell side only for delivery;
#                on both sides for F&O / intraday)
#    GST      — 18% on brokerage + exchange charges
#    Stamp    — Stamp duty on buy side (state-level, varies; using 0.015%)
#    Exchange — NSE/BSE exchange transaction charges
#    SEBI     — SEBI turnover fee
#    DP       — Depository participant charge (for CNC/delivery, skip intraday)
#
#  Rates as of FY2025 (NSE cash market — CNC/MIS).
#  For F&O (futures/options), call with trade_type="FO".
#
#  Reference:
#    https://zerodha.com/charges
#    https://www.nseindia.com/regulations/member-regulation-transaction-charges
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

from dataclasses import dataclass


# ── Rate constants (FY2025, NSE cash market) ─────────────────────

# STT — Securities Transaction Tax
# Intraday equity (MIS): 0.025% on sell side only
# Delivery equity (CNC): 0.1% on both buy and sell
# Equity futures: 0.0125% on sell side
# Equity options: 0.0625% on sell side (on premium)
STT_INTRADAY_PCT  = 0.025 / 100    # sell side only
STT_DELIVERY_PCT  = 0.100 / 100    # both sides
STT_FUTURES_PCT   = 0.0125 / 100   # sell side
STT_OPTIONS_PCT   = 0.0625 / 100   # on premium, sell side

# NSE Exchange Transaction Charges (on total turnover)
EXCHANGE_EQUITY_PCT  = 0.00322 / 100   # 0.00322% cash
EXCHANGE_FUTURES_PCT = 0.00188 / 100   # 0.00188% futures
EXCHANGE_OPTIONS_PCT = 0.05309 / 100   # 0.05309% on premium

# SEBI turnover fee (same for all segments)
SEBI_FEE_PCT = 0.0001 / 100   # ₹10 per crore turnover = 0.00001%

# Stamp duty (buyer only — state-level average for Maharashtra)
STAMP_INTRADAY_PCT = 0.003 / 100     # 0.003% on buy side
STAMP_DELIVERY_PCT = 0.015 / 100     # 0.015% on buy side
STAMP_FUTURES_PCT  = 0.002 / 100     # 0.002% on buy side
STAMP_OPTIONS_PCT  = 0.003 / 100     # 0.003% on buy side (premium only)

# Brokerage — typical discount broker flat fee per trade
BROKERAGE_FLAT_INR = 20.0           # ₹20 per trade (Zerodha/Dhan model)
BROKERAGE_MAX_PCT  = 0.25 / 100     # but never more than 0.25% of turnover

# GST on (brokerage + exchange charges)
GST_PCT = 0.18

# Depository participant charge per sell trade (CDSL/NSDL)
DP_CHARGE_INR = 13.5   # ₹13.5 flat per ISIN per sell (for CNC only)


@dataclass
class TradeCost:
    """
    Itemised breakdown of transaction costs for a single trade leg.
    All values in INR.
    """
    turnover: float         # total buy or sell value
    brokerage: float
    stt: float
    exchange_charge: float
    sebi_fee: float
    stamp_duty: float
    gst: float
    dp_charge: float

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.exchange_charge
            + self.sebi_fee
            + self.stamp_duty
            + self.gst
            + self.dp_charge
        )

    @property
    def total_pct(self) -> float:
        """Total cost as % of turnover."""
        return (self.total / self.turnover * 100) if self.turnover else 0.0


def calculate_round_trip_cost(
    entry_price: float,
    exit_price:  float,
    quantity:    float,
    trade_type:  str = "MIS",   # "MIS" | "CNC" | "FO_FUTURES" | "FO_OPTIONS"
) -> tuple[TradeCost, TradeCost]:
    """
    Calculate per-leg costs for a round-trip trade (buy + sell).

    Args:
        entry_price: Buy price per unit (₹)
        exit_price:  Sell price per unit (₹)
        quantity:    Number of shares / lots
        trade_type:  "MIS" (intraday), "CNC" (delivery), "FO_FUTURES", "FO_OPTIONS"

    Returns:
        (buy_cost, sell_cost) — each a TradeCost dataclass.
    """
    buy_turnover  = entry_price * quantity
    sell_turnover = exit_price  * quantity

    if trade_type == "MIS":
        buy_cost  = _equity_cost(buy_turnover,  side="buy",  delivery=False, is_options=False)
        sell_cost = _equity_cost(sell_turnover, side="sell", delivery=False, is_options=False)
    elif trade_type == "CNC":
        buy_cost  = _equity_cost(buy_turnover,  side="buy",  delivery=True, is_options=False)
        sell_cost = _equity_cost(sell_turnover, side="sell", delivery=True, is_options=False)
    elif trade_type == "FO_FUTURES":
        buy_cost  = _fo_cost(buy_turnover,  side="buy",  is_options=False)
        sell_cost = _fo_cost(sell_turnover, side="sell", is_options=False)
    elif trade_type == "FO_OPTIONS":
        buy_cost  = _fo_cost(buy_turnover,  side="buy",  is_options=True)
        sell_cost = _fo_cost(sell_turnover, side="sell", is_options=True)
    else:
        raise ValueError(f"Unknown trade_type: {trade_type!r}")

    return buy_cost, sell_cost


def net_pnl_after_costs(
    entry_price: float,
    exit_price:  float,
    quantity:    float,
    direction:   str  = "LONG",
    trade_type:  str  = "MIS",
) -> tuple[float, float]:
    """
    Compute gross PnL and net PnL (after all transaction costs).

    Args:
        direction: "LONG" or "SHORT"
        trade_type: See calculate_round_trip_cost.

    Returns:
        (gross_pnl, net_pnl) in INR
    """
    if direction == "LONG":
        gross = (exit_price - entry_price) * quantity
    else:
        gross = (entry_price - exit_price) * quantity

    buy_cost, sell_cost = calculate_round_trip_cost(
        entry_price, exit_price, quantity, trade_type
    )
    net = gross - buy_cost.total - sell_cost.total
    return round(gross, 2), round(net, 2)


def effective_cost_pct(
    entry_price: float,
    exit_price:  float,
    quantity:    float,
    trade_type:  str = "MIS",
) -> float:
    """
    Total round-trip cost as % of mean turnover.
    Useful for comparing strategies or slippage budgeting.
    """
    buy_cost, sell_cost = calculate_round_trip_cost(
        entry_price, exit_price, quantity, trade_type
    )
    mean_turnover = (entry_price + exit_price) * quantity / 2
    if mean_turnover == 0:
        return 0.0
    return round((buy_cost.total + sell_cost.total) / mean_turnover * 100, 4)


# ── Internal helpers ──────────────────────────────────────────────

def _brokerage(turnover: float) -> float:
    """Flat ₹20 per trade, capped at 0.25% of turnover."""
    return min(BROKERAGE_FLAT_INR, turnover * BROKERAGE_MAX_PCT)


def _equity_cost(turnover: float, side: str, delivery: bool, is_options: bool) -> TradeCost:
    brokerage = 0.0 if delivery else _brokerage(turnover)
    exchange  = turnover * EXCHANGE_EQUITY_PCT
    sebi      = turnover * SEBI_FEE_PCT
    gst_base  = brokerage + exchange
    gst       = gst_base * GST_PCT

    if delivery:
        stt   = turnover * STT_DELIVERY_PCT
        stamp = (turnover * STAMP_DELIVERY_PCT) if side == "buy" else 0.0
        dp    = DP_CHARGE_INR if side == "sell" else 0.0
    else:
        stt   = (turnover * STT_INTRADAY_PCT) if side == "sell" else 0.0
        stamp = (turnover * STAMP_INTRADAY_PCT) if side == "buy" else 0.0
        dp    = 0.0

    return TradeCost(
        turnover=turnover,
        brokerage=round(brokerage, 4),
        stt=round(stt, 4),
        exchange_charge=round(exchange, 4),
        sebi_fee=round(sebi, 6),
        stamp_duty=round(stamp, 4),
        gst=round(gst, 4),
        dp_charge=round(dp, 4),
    )


def _fo_cost(turnover: float, side: str, is_options: bool) -> TradeCost:
    brokerage = _brokerage(turnover)
    if is_options:
        exchange = turnover * EXCHANGE_OPTIONS_PCT
        stt      = (turnover * STT_OPTIONS_PCT) if side == "sell" else 0.0
        stamp    = (turnover * STAMP_OPTIONS_PCT) if side == "buy" else 0.0
    else:
        exchange = turnover * EXCHANGE_FUTURES_PCT
        stt      = (turnover * STT_FUTURES_PCT) if side == "sell" else 0.0
        stamp    = (turnover * STAMP_FUTURES_PCT) if side == "buy" else 0.0

    sebi = turnover * SEBI_FEE_PCT
    gst  = (brokerage + exchange) * GST_PCT

    return TradeCost(
        turnover=turnover,
        brokerage=round(brokerage, 4),
        stt=round(stt, 4),
        exchange_charge=round(exchange, 4),
        sebi_fee=round(sebi, 6),
        stamp_duty=round(stamp, 4),
        gst=round(gst, 4),
        dp_charge=0.0,    # no DP for F&O
    )
