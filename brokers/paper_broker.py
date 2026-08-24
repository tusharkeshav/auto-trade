# ─────────────────────────────────────────────────────────────────
#  brokers/paper_broker.py
#  In-process paper broker for India market.
#
#  Simulates fills at current market price (no slippage model —
#  same as the BTC paper trader). Tracks positions and balance
#  independently of engine/portfolio.py so it can be swapped for
#  a live broker without changing the trading loop.
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import uuid
from typing import Optional

from loguru import logger

from brokers.base_broker import BaseBroker, BrokerOrderResult, BrokerPosition
from data.india.nse_client import NSEClient


class PaperBrokerIndia(BaseBroker):
    """
    Simulated paper broker for NSE/BSE instruments.

    Fills market orders immediately at the last yfinance quote.
    Limit orders are stored and checked on next get_quote() call.
    """

    def __init__(self, initial_balance_inr: float = 500_000.0):
        self._balance:   float                        = initial_balance_inr
        self._positions: dict[str, BrokerPosition]    = {}   # symbol → position
        self._client:    NSEClient                    = NSEClient()

    # ── BaseBroker interface ──────────────────────────────────────

    @property
    def name(self) -> str:
        return "paper_india"

    def get_quote(self, symbol: str) -> float:
        return self._client.get_price(symbol)

    def place_market_order(
        self,
        symbol:    str,
        direction: str,
        qty:       float,
        product:   str = "MIS",
    ) -> BrokerOrderResult:
        price = self.get_quote(symbol)
        order_id = str(uuid.uuid4())[:8]

        if direction == "BUY":
            cost = price * qty
            if cost > self._balance:
                return BrokerOrderResult(
                    success=False, order_id=order_id, symbol=symbol,
                    qty=qty, price=price, direction=direction,
                    order_type="MARKET", status="REJECTED",
                    message=f"Insufficient balance: ₹{self._balance:,.2f} < ₹{cost:,.2f}",
                )
            self._balance -= cost
            if symbol in self._positions:
                # Average into existing long
                pos = self._positions[symbol]
                total_qty = pos.qty + qty
                avg = (pos.avg_price * pos.qty + price * qty) / total_qty
                self._positions[symbol] = BrokerPosition(
                    symbol=symbol, qty=total_qty, avg_price=avg,
                    current_price=price, pnl=0.0, direction="LONG",
                )
            else:
                self._positions[symbol] = BrokerPosition(
                    symbol=symbol, qty=qty, avg_price=price,
                    current_price=price, pnl=0.0, direction="LONG",
                )
            logger.success(f"[paper] BUY {qty} {symbol} @ ₹{price:,.2f} | balance ₹{self._balance:,.2f}")

        else:  # SELL
            if symbol not in self._positions:
                return BrokerOrderResult(
                    success=False, order_id=order_id, symbol=symbol,
                    qty=qty, price=price, direction=direction,
                    order_type="MARKET", status="REJECTED",
                    message=f"No open position for {symbol}",
                )
            pos      = self._positions[symbol]
            fill_qty = min(qty, pos.qty)   # can't sell more than held
            self._balance += price * fill_qty
            remaining  = pos.qty - fill_qty
            if remaining <= 0:
                del self._positions[symbol]
            else:
                self._positions[symbol] = BrokerPosition(
                    symbol=symbol, qty=remaining, avg_price=pos.avg_price,
                    current_price=price,
                    pnl=(price - pos.avg_price) * remaining,
                    direction="LONG",
                )
            logger.success(f"[paper] SELL {fill_qty} {symbol} @ ₹{price:,.2f} | balance ₹{self._balance:,.2f}")
            return BrokerOrderResult(
                success=True, order_id=order_id, symbol=symbol,
                qty=fill_qty, price=price, direction=direction,
                order_type="MARKET", status="FILLED",
            )

        # BUY path — return here
        return BrokerOrderResult(
            success=True, order_id=order_id, symbol=symbol,
            qty=qty, price=price, direction=direction,
            order_type="MARKET", status="FILLED",
        )

    def place_limit_order(
        self,
        symbol:      str,
        direction:   str,
        qty:         float,
        limit_price: float,
        product:     str = "MIS",
    ) -> BrokerOrderResult:
        # Paper broker: check immediately against current quote
        current = self.get_quote(symbol)
        if (direction == "BUY" and current <= limit_price) or \
           (direction == "SELL" and current >= limit_price):
            # Fill immediately at limit price
            return self.place_market_order(symbol, direction, qty, product)

        # Not filled yet — return PLACED (caller must poll)
        logger.info(f"[paper] LIMIT {direction} {qty} {symbol} @ ₹{limit_price:,.2f} (current ₹{current:,.2f}) — not filled")
        return BrokerOrderResult(
            success=True, order_id=str(uuid.uuid4())[:8], symbol=symbol,
            qty=qty, price=limit_price, direction=direction,
            order_type="LIMIT", status="PLACED",
        )

    def cancel_order(self, order_id: str) -> BrokerOrderResult:
        # Paper broker: limit orders are not queued, nothing to cancel
        return BrokerOrderResult(
            success=True, order_id=order_id, symbol="", qty=0, price=0,
            direction="", order_type="", status="CANCELLED",
        )

    def get_positions(self) -> list[BrokerPosition]:
        # Refresh current_price and pnl
        updated = []
        for sym, pos in self._positions.items():
            try:
                price = self.get_quote(sym)
                pnl   = (price - pos.avg_price) * pos.qty
                updated.append(BrokerPosition(
                    symbol=sym, qty=pos.qty, avg_price=pos.avg_price,
                    current_price=price, pnl=pnl, direction=pos.direction,
                ))
            except Exception:
                updated.append(pos)
        return updated

    def get_balance(self) -> float:
        return self._balance
