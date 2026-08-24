# ─────────────────────────────────────────────────────────────────
#  engine/portfolio.py
#  In-memory portfolio state — positions, cash, and P&L tracking.
#  Designed for speed: all operations are O(1) dict lookups.
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime    import datetime, timezone
from typing      import Optional

from config.settings import INITIAL_CAPITAL_USDT, MAX_OPEN_TRADES, MAX_RISK_PER_TRADE_PCT, MAX_POSITION_PCT


# ─────────────────────────────────────────────────────────────────
#  Position
# ─────────────────────────────────────────────────────────────────

@dataclass
class Position:
    """
    A single open paper trade.
    All-In All-Out: full exit at either SL or TP (1.5R).
    """
    id:           str
    symbol:       str
    direction:    str       # "LONG" | "SHORT"

    entry_price:  float
    size_total:   float     # total units (e.g. BTC)
    size_remaining: float   # units still open

    stop_loss:    float     # stop-hunt-safe (1 ATR below support)
    take_profit1: float     # unused in AIAO but kept for signal compatibility
    take_profit2: float     # single TP exit at 1.5R

    probability:  float     # probability score at entry time
    entry_time:   datetime  = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    def unrealized_pnl(self, current_price: float) -> float:
        if self.direction == "LONG":
            return (current_price - self.entry_price) * self.size_remaining
        return (self.entry_price - current_price) * self.size_remaining

    def unrealized_pnl_pct(self, current_price: float) -> float:
        pnl = self.unrealized_pnl(current_price)
        cost = self.entry_price * self.size_remaining
        return (pnl / cost) * 100 if cost else 0.0

    def is_sl_hit(self, current_price: float) -> bool:
        if self.direction == "LONG":  return current_price <= self.stop_loss
        return current_price >= self.stop_loss

    def is_tp2_hit(self, current_price: float) -> bool:
        if self.direction == "LONG":  return current_price >= self.take_profit2
        return current_price <= self.take_profit2


# ─────────────────────────────────────────────────────────────────
#  Portfolio
# ─────────────────────────────────────────────────────────────────

class Portfolio:
    """
    Tracks cash, open positions, and cumulative performance.
    All state is held in-memory for maximum speed.
    """

    def __init__(self, initial_capital: float = INITIAL_CAPITAL_USDT):
        self.initial_capital:  float                    = initial_capital
        self.cash:             float                    = initial_capital
        self.positions:        dict[str, Position]      = {}   # id → Position
        self.total_trades:     int                      = 0
        self.winning_trades:   int                      = 0
        self.total_realized_pnl: float                  = 0.0

    # ── Computed properties ───────────────────────────────────────

    def total_value(self, prices: dict[str, float]) -> float:
        """Cash + market value of all open positions."""
        position_value = sum(
            p.size_remaining * prices.get(p.symbol, p.entry_price)
            for p in self.positions.values()
        )
        return self.cash + position_value

    def total_pnl(self, prices: dict[str, float]) -> float:
        return self.total_value(prices) - self.initial_capital

    def total_pnl_pct(self, prices: dict[str, float]) -> float:
        return (self.total_pnl(prices) / self.initial_capital) * 100

    def win_rate(self) -> float:
        if self.total_trades == 0: return 0.0
        return (self.winning_trades / self.total_trades) * 100

    def open_position_count(self) -> int:
        return len(self.positions)

    def can_open_trade(self) -> bool:
        """Gate check: respects max simultaneous positions."""
        return self.open_position_count() < MAX_OPEN_TRADES

    # ── Position sizing ───────────────────────────────────────────

    def calculate_position_size(self, entry: float, stop_loss: float) -> float:
        """
        Fixed fractional position sizing with capital cap.

        Step 1 — Risk budget: risk exactly MAX_RISK_PER_TRADE_PCT% per trade.
            size = (cash × risk_pct) / |entry − stop_loss|

        Step 2 — Capital cap: never invest more than MAX_POSITION_PCT% of
            cash in one position (prevents over-leverage on expensive assets
            like BTC where a single unit costs more than the account).
        """
        risk_budget   = self.cash * (MAX_RISK_PER_TRADE_PCT / 100)
        risk_per_unit = abs(entry - stop_loss)
        if risk_per_unit == 0:
            return 0.0

        size_by_risk = risk_budget / risk_per_unit

        # Cap: never invest more than MAX_POSITION_PCT% of cash in one trade
        max_affordable = (self.cash * MAX_POSITION_PCT / 100) / entry
        size = min(size_by_risk, max_affordable)

        return round(size, 6)

    # ── Position management ───────────────────────────────────────

    def open_position(self, signal) -> Optional[Position]:
        """
        Open a new position from a TradeSignal.
        Returns None if insufficient funds or max trades reached.
        """
        if not self.can_open_trade():
            return None

        size = self.calculate_position_size(signal.entry_price, signal.stop_loss)
        cost = size * signal.entry_price

        if cost > self.cash or size <= 0:
            return None

        position = Position(
            id            = str(uuid.uuid4())[:8],
            symbol        = signal.symbol,
            direction     = signal.direction,
            entry_price   = signal.entry_price,
            size_total    = size,
            size_remaining= size,
            stop_loss     = signal.stop_loss,
            take_profit1  = signal.take_profit1,
            take_profit2  = signal.take_profit2,
            probability   = signal.probability,
        )

        self.cash -= cost
        self.positions[position.id] = position
        return position

    def close_position(self, position: Position, exit_price: float) -> float:
        """Fully close a position. Returns total realized P&L."""
        if position.direction == "LONG":
            pnl = (exit_price - position.entry_price) * position.size_remaining
        else:
            pnl = (position.entry_price - exit_price) * position.size_remaining

        self.cash += position.size_remaining * exit_price
        self.total_realized_pnl += pnl
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        del self.positions[position.id]
        return round(pnl, 4)
