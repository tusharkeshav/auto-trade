# ─────────────────────────────────────────────────────────────────
#  engine/order_manager.py
#  Simulates order execution and monitors open positions.
#
#  Key rules enforced here:
#    1. No duplicate symbol positions (one position per symbol)
#    2. All-In All-Out: single SL or TP exit, no partials
#    3. Consecutive-check guard: SL must be hit on 2 checks in a row
#       to prevent single-tick wicks from stopping us out (stop hunt guard)
#    4. Daily loss limit: blocks new entries when daily losses exceed threshold
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import uuid
from datetime    import datetime, timezone

from loguru import logger

from engine.portfolio import Portfolio, Position
from engine.ledger    import Ledger, TradeRecord
from probability.signal_scorer import TradeSignal
from config.settings  import DAILY_LOSS_LIMIT_PCT


class OrderManager:
    """
    Manages the lifecycle of paper trades:
      open → monitor → close (TP or SL)
    """

    def __init__(self, portfolio: Portfolio, ledger: Ledger):
        self.portfolio = portfolio
        self.ledger    = ledger
        self._sl_check_count: dict[str, int] = {}   # position_id → consecutive SL hit count
        self._daily_realized_pnl: float = 0.0
        self._daily_pnl_date: str = ""  # YYYY-MM-DD of current tracking day

    # ─────────────────────────────────────────────────────────────
    #  Open a new position
    # ─────────────────────────────────────────────────────────────

    def _reset_daily_pnl_if_new_day(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_pnl_date:
            self._daily_realized_pnl = 0.0
            self._daily_pnl_date = today

    def is_daily_loss_limit_hit(self) -> bool:
        self._reset_daily_pnl_if_new_day()
        limit = self.portfolio.initial_capital * (DAILY_LOSS_LIMIT_PCT / 100)
        return self._daily_realized_pnl <= -limit

    def try_open(self, signal: TradeSignal) -> Position | None:
        """
        Attempt to open a position from a TradeSignal.

        Guards:
          - Signal must be tradeable (LONG or SHORT)
          - Portfolio must not already have an open position in this symbol
          - Portfolio must not exceed MAX_OPEN_TRADES
          - Daily loss limit must not be breached
        """
        if not signal.is_tradeable():
            return None

        if self.is_daily_loss_limit_hit():
            logger.warning(
                f"Daily loss limit hit (${self._daily_realized_pnl:,.2f}) — "
                f"blocking new entries until next day"
            )
            return None

        open_symbols = {p.symbol for p in self.portfolio.positions.values()}
        if signal.symbol in open_symbols:
            logger.debug(f"Already have open position in {signal.symbol} — skipping")
            return None

        position = self.portfolio.open_position(signal)
        if position is None:
            logger.warning(f"Could not open position for {signal.symbol} — check capital/limits")
            return None

        logger.success(
            f"📈 OPENED {position.direction} {signal.symbol}  |  "
            f"Entry: ${position.entry_price:,.2f}  |  "
            f"Size: {position.size_total:.6f}  |  "
            f"SL: ${position.stop_loss:,.2f}  |  "
            f"TP: ${position.take_profit2:,.2f}  |  "
            f"Probability: {signal.probability:.1f}%"
        )
        return position

    # ─────────────────────────────────────────────────────────────
    #  Monitor all open positions against current prices
    # ─────────────────────────────────────────────────────────────

    def check_positions(self, prices: dict[str, float]) -> None:
        """
        Called every loop iteration with latest prices.
        All-In, All-Out: closes the ENTIRE position at either SL or TP (1.5R).
        There are no partial exits or breakeven moves.
        """
        for pos_id, position in list(self.portfolio.positions.items()):
            price = prices.get(position.symbol)
            if price is None:
                continue

            # ── Stop Loss (with consecutive-check guard against wick stops) ──────
            if position.is_sl_hit(price):
                count = self._sl_check_count.get(pos_id, 0) + 1
                self._sl_check_count[pos_id] = count

                if count >= 2:
                    self._close_full(position, price, "STOP_LOSS")
                    self._sl_check_count.pop(pos_id, None)
                else:
                    logger.debug(
                        f"⚠️  SL check {count}/2 for {position.symbol} @ ${price:,.2f} "
                        f"(SL: ${position.stop_loss:,.2f}) — waiting for confirmation"
                    )
            else:
                self._sl_check_count.pop(pos_id, None)

            # ── Take Profit (single full exit at 1.5R) ────────────────────
            if pos_id in self.portfolio.positions:   # may have been removed by SL above
                if position.is_tp2_hit(price):
                    self._close_full(position, price, "TAKE_PROFIT")

    # ─────────────────────────────────────────────────────────────
    #  Internal close helpers
    # ─────────────────────────────────────────────────────────────

    def _close_full(self, position: Position, exit_price: float, close_type: str) -> None:
        """Close the entire position. Tracks daily realized PnL."""
        pnl = self.portfolio.close_position(position, exit_price)

        self._reset_daily_pnl_if_new_day()
        self._daily_realized_pnl += pnl

        self._record_trade(position, exit_price, pnl, close_type, fraction=1.0)

        icon  = "🎯" if "PROFIT" in close_type else "🛑"
        label = {"TAKE_PROFIT": "TP HIT", "STOP_LOSS": "SL HIT"}.get(close_type, close_type)
        logger.info(
            f"{icon} {label} {position.symbol}  |  "
            f"Exit: ${exit_price:,.2f}  |  "
            f"P&L: {'+' if pnl >= 0 else ''}${pnl:,.2f}  |  "
            f"{'WIN ✅' if pnl > 0 else 'LOSS ❌'}  |  "
            f"Daily P&L: ${self._daily_realized_pnl:,.2f}"
        )

    def _record_trade(
        self,
        position:   Position,
        exit_price: float,
        pnl:        float,
        close_type: str,
        fraction:   float,
    ) -> None:
        size_closed = position.size_total * fraction
        pnl_pct     = (pnl / (position.entry_price * size_closed)) * 100 if size_closed else 0.0

        self.ledger.record(TradeRecord(
            trade_id    = str(uuid.uuid4())[:8],
            position_id = position.id,
            symbol      = position.symbol,
            direction   = position.direction,
            close_type  = close_type,
            entry_price = position.entry_price,
            exit_price  = exit_price,
            size        = size_closed,
            pnl         = round(pnl, 4),
            pnl_pct     = round(pnl_pct, 4),
            probability = position.probability,
            entry_time  = position.entry_time,
            exit_time   = datetime.now(timezone.utc),
        ))
