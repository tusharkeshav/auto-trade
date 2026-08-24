# ─────────────────────────────────────────────────────────────────
#  engine/paper_orchestrator_db.py
#  ACID SQLite Database Persistence for AI Meta-Orchestrator Paper Trading.
# ─────────────────────────────────────────────────────────────────

import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
DB_PATH = Path(__file__).resolve().parent.parent / "ai_meta_paper_portfolio.sqlite"


class PaperOrchestratorDB:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_tables()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        with self._get_conn() as conn:
            cur = conn.cursor()
            # 1. Portfolio State
            cur.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_state (
                id INTEGER PRIMARY KEY,
                cash_balance_inr REAL NOT NULL,
                invested_inr REAL NOT NULL,
                current_nav_inr REAL NOT NULL,
                peak_nav_inr REAL NOT NULL,
                realized_pnl_inr REAL NOT NULL,
                total_taxes_paid_inr REAL NOT NULL,
                active_regime TEXT NOT NULL,
                last_run_date TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """)

            # 2. Open Positions
            cur.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                strategy_name TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                current_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                risk_unit REAL NOT NULL,
                be_locked INTEGER NOT NULL DEFAULT 0,
                unrealized_pnl REAL NOT NULL,
                unrealized_pnl_pct REAL NOT NULL
            )
            """)

            # 3. Trade History
            cur.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                exit_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                gross_pnl REAL NOT NULL,
                taxes_inr REAL NOT NULL,
                net_pnl REAL NOT NULL,
                net_pnl_pct REAL NOT NULL,
                exit_reason TEXT NOT NULL,
                bars_held INTEGER NOT NULL
            )
            """)

            # 4. Daily NAV History
            cur.execute("""
            CREATE TABLE IF NOT EXISTS nav_history (
                date TEXT PRIMARY KEY,
                nav_inr REAL NOT NULL,
                cash_inr REAL NOT NULL,
                invested_inr REAL NOT NULL,
                benchmark_price REAL,
                regime TEXT NOT NULL
            )
            """)

            # Initialize Default Portfolio State if empty
            cur.execute("SELECT COUNT(*) FROM portfolio_state")
            if cur.fetchone()[0] == 0:
                now_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
                today_str = datetime.now(IST).strftime("%Y-%m-%d")
                cur.execute("""
                INSERT INTO portfolio_state (
                    id, cash_balance_inr, invested_inr, current_nav_inr,
                    peak_nav_inr, realized_pnl_inr, total_taxes_paid_inr,
                    active_regime, last_run_date, created_at, updated_at
                ) VALUES (1, 100000.0, 0.0, 100000.0, 100000.0, 0.0, 0.0, 'CHOPPY_SIDEWAYS', ?, ?, ?)
                """, (today_str, now_str, now_str))
            conn.commit()

    def get_portfolio_state(self) -> Dict[str, Any]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM portfolio_state WHERE id = 1")
            row = cur.fetchone()
            return dict(row) if row else {}

    def update_portfolio_state(self, cash: float, invested: float, nav: float, realized_pnl: float, taxes: float, regime: str, last_date: str):
        now_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        state = self.get_portfolio_state()
        peak = max(state.get("peak_nav_inr", nav), nav)

        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
            UPDATE portfolio_state SET
                cash_balance_inr = ?,
                invested_inr = ?,
                current_nav_inr = ?,
                peak_nav_inr = ?,
                realized_pnl_inr = ?,
                total_taxes_paid_inr = ?,
                active_regime = ?,
                last_run_date = ?,
                updated_at = ?
            WHERE id = 1
            """, (cash, invested, nav, peak, realized_pnl, taxes, regime, last_date, now_str))
            conn.commit()

    def get_open_positions(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM positions")
            return [dict(r) for r in cur.fetchall()]

    def add_position(self, pos: Dict[str, Any]):
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT OR REPLACE INTO positions (
                symbol, strategy_name, entry_date, entry_price, quantity,
                current_price, stop_loss, take_profit, risk_unit, be_locked,
                unrealized_pnl, unrealized_pnl_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pos["symbol"], pos["strategy_name"], pos["entry_date"], pos["entry_price"],
                pos["quantity"], pos["current_price"], pos["stop_loss"], pos["take_profit"],
                pos["risk_unit"], 1 if pos.get("be_locked") else 0,
                pos.get("unrealized_pnl", 0.0), pos.get("unrealized_pnl_pct", 0.0)
            ))
            conn.commit()

    def update_position_price(self, symbol: str, current_price: float, be_locked: bool = False, stop_loss: Optional[float] = None):
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT entry_price, quantity, stop_loss FROM positions WHERE symbol = ?", (symbol,))
            row = cur.fetchone()
            if not row: return

            e_px, qty, sl_old = row["entry_price"], row["quantity"], row["stop_loss"]
            sl_new = stop_loss if stop_loss is not None else sl_old
            unreal_pnl = (current_price - e_px) * qty
            unreal_pct = ((current_price - e_px) / e_px) * 100.0 if e_px > 0 else 0.0

            cur.execute("""
            UPDATE positions SET
                current_price = ?,
                stop_loss = ?,
                be_locked = ?,
                unrealized_pnl = ?,
                unrealized_pnl_pct = ?
            WHERE symbol = ?
            """, (current_price, sl_new, 1 if be_locked else 0, unreal_pnl, unreal_pct, symbol))
            conn.commit()

    def close_position(self, symbol: str, exit_price: float, exit_date: str, exit_reason: str, taxes_inr: float) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM positions WHERE symbol = ?", (symbol,))
            row = cur.fetchone()
            if not row: return None

            pos = dict(row)
            gross_pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
            net_pnl = gross_pnl - taxes_inr
            invested = pos["entry_price"] * pos["quantity"]
            net_pct = (net_pnl / invested) * 100.0 if invested > 0 else 0.0

            # Calculate days held
            try:
                d_entry = datetime.strptime(pos["entry_date"][:10], "%Y-%m-%d")
                d_exit  = datetime.strptime(exit_date[:10], "%Y-%m-%d")
                bars_held = max(1, (d_exit - d_entry).days)
            except Exception:
                bars_held = 1

            # Log Trade History
            cur.execute("""
            INSERT INTO trade_history (
                symbol, strategy_name, entry_date, exit_date, entry_price,
                exit_price, quantity, gross_pnl, taxes_inr, net_pnl,
                net_pnl_pct, exit_reason, bars_held
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pos["symbol"], pos["strategy_name"], pos["entry_date"], exit_date,
                pos["entry_price"], exit_price, pos["quantity"], gross_pnl,
                taxes_inr, net_pnl, net_pct, exit_reason, bars_held
            ))

            # Delete from positions
            cur.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            conn.commit()

            return {
                "symbol": symbol,
                "strategy": pos["strategy_name"],
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "quantity": pos["quantity"],
                "gross_pnl": gross_pnl,
                "taxes_inr": taxes_inr,
                "net_pnl": net_pnl,
                "net_pnl_pct": net_pct,
                "exit_reason": exit_reason,
            }

    def log_daily_nav(self, date_str: str, nav: float, cash: float, invested: float, bm_price: float, regime: str):
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT OR REPLACE INTO nav_history (
                date, nav_inr, cash_inr, invested_inr, benchmark_price, regime
            ) VALUES (?, ?, ?, ?, ?, ?)
            """, (date_str, nav, cash, invested, bm_price, regime))
            conn.commit()

    def get_trade_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM trade_history ORDER BY trade_id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    def reset_portfolio(self, initial_capital: float = 100000.0):
        now_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        today_str = datetime.now(IST).strftime("%Y-%m-%d")
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM positions")
            cur.execute("DELETE FROM trade_history")
            cur.execute("DELETE FROM nav_history")
            cur.execute("""
            UPDATE portfolio_state SET
                cash_balance_inr = ?,
                invested_inr = 0.0,
                current_nav_inr = ?,
                peak_nav_inr = ?,
                realized_pnl_inr = 0.0,
                total_taxes_paid_inr = 0.0,
                active_regime = 'CHOPPY_SIDEWAYS',
                last_run_date = ?,
                updated_at = ?
            WHERE id = 1
            """, (initial_capital, initial_capital, initial_capital, today_str, now_str))
            conn.commit()
