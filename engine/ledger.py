# ─────────────────────────────────────────────────────────────────
#  engine/ledger.py
#  Immutable trade log.
#  Stores every completed trade in memory and flushes to Parquet
#  on demand. Use DuckDB to query for analytics and backtesting.
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime    import datetime
from pathlib     import Path

import pandas as pd

LEDGER_PATH = Path("data/ledger")


@dataclass(frozen=True)
class TradeRecord:
    """Immutable record of a completed (or partially completed) trade."""
    trade_id:    str
    position_id: str
    symbol:      str
    direction:   str       # LONG | SHORT
    close_type:  str       # STOP_LOSS | TAKE_PROFIT_1 | TAKE_PROFIT_2 | MANUAL
    entry_price: float
    exit_price:  float
    size:        float     # units closed
    pnl:         float     # realized P&L in USDT
    pnl_pct:     float
    probability: float     # signal probability at entry
    entry_time:  datetime
    exit_time:   datetime
    interval:    str = "15m"  # "15m" or "30m" — which timeframe generated the signal

    @property
    def is_win(self) -> bool:
        return self.pnl > 0

    @property
    def result_label(self) -> str:
        return "✅ WIN" if self.is_win else "❌ LOSS"


class Ledger:
    """
    Append-only trade log.

    Usage:
        ledger.record(trade_record)
        ledger.save()                      # flush to Parquet
        df = ledger.to_dataframe()         # query with DuckDB
    """

    def __init__(self):
        self._records: list[TradeRecord] = []
        LEDGER_PATH.mkdir(parents=True, exist_ok=True)

    def record(self, trade: TradeRecord) -> None:
        self._records.append(trade)

    def recent(self, n: int = 10) -> list[TradeRecord]:
        return self._records[-n:]

    def to_dataframe(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame([vars(r) for r in self._records])

    def total_pnl(self) -> float:
        return sum(r.pnl for r in self._records)

    def win_count(self) -> int:
        return sum(1 for r in self._records if r.is_win)

    def loss_count(self) -> int:
        return sum(1 for r in self._records if not r.is_win)

    def win_rate(self) -> float:
        total = len(self._records)
        return (self.win_count() / total * 100) if total else 0.0

    def save(self) -> Path:
        """Flush all records to a file (Parquet if pyarrow available, else CSV)."""
        df = self.to_dataframe()
        if df.empty:
            return LEDGER_PATH

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        try:
            import pyarrow  # noqa: F401
            path = LEDGER_PATH / f"trades_{ts}.parquet"
            df.to_parquet(path, index=False)
        except ImportError:
            path = LEDGER_PATH / f"trades_{ts}.csv"
            df.to_csv(path, index=False)

        return path
