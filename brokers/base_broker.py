# ─────────────────────────────────────────────────────────────────
#  brokers/base_broker.py
#  Abstract broker interface — paper and live implementations
#  must conform to this contract.
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

from abc      import ABC, abstractmethod
from dataclasses import dataclass
from datetime    import datetime
from typing      import Optional


@dataclass
class BrokerOrderResult:
    """Returned by place_order / cancel_order."""
    success:      bool
    order_id:     str
    symbol:       str
    qty:          float
    price:        float         # filled price (0.0 if not yet filled)
    direction:    str           # "BUY" | "SELL"
    order_type:   str           # "MARKET" | "LIMIT"
    status:       str           # "PLACED" | "FILLED" | "REJECTED" | "CANCELLED"
    message:      str = ""      # error / info message from broker


@dataclass
class BrokerPosition:
    """Current open position as reported by broker."""
    symbol:       str
    qty:          float
    avg_price:    float
    current_price: float
    pnl:          float
    direction:    str           # "LONG" | "SHORT"


class BaseBroker(ABC):
    """
    Abstract broker interface.

    Paper implementations simulate fills in-process.
    Live implementations call real broker REST APIs.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Broker/mode identifier string e.g. 'paper', 'dhan_sandbox'."""

    @abstractmethod
    def get_quote(self, symbol: str) -> float:
        """Return latest LTP (last traded price) for the symbol."""

    @abstractmethod
    def place_market_order(
        self,
        symbol:    str,
        direction: str,    # "BUY" | "SELL"
        qty:       float,
        product:   str = "MIS",   # "MIS" | "CNC" | "NRML"
    ) -> BrokerOrderResult:
        """Execute a market order. Returns the filled result."""

    @abstractmethod
    def place_limit_order(
        self,
        symbol:    str,
        direction: str,
        qty:       float,
        limit_price: float,
        product:   str = "MIS",
    ) -> BrokerOrderResult:
        """Place a limit order. May not fill immediately."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> BrokerOrderResult:
        """Cancel an open order."""

    @abstractmethod
    def get_positions(self) -> list[BrokerPosition]:
        """Return all currently open positions."""

    @abstractmethod
    def get_balance(self) -> float:
        """Return available cash balance in INR."""
