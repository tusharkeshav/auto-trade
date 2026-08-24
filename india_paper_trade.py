# ─────────────────────────────────────────────────────────────────
#  india_paper_trade.py
#
#  Production Paper Trading Daemon for India Equities.
#  100% Isolated from Crypto / Binance code.
#
#  Data Source : NSEClient (yfinance) — 100% Free.
#  Strategy    : Master Spot Index → Stock CNC Delivery (Rule 2).
#  Execution   : SQLite State (ACID compliant local simulated broker).
#
#  Usage:
#      python india_paper_trade.py               # Run once immediately
#      python india_paper_trade.py --daemon      # Run forever in background
# ─────────────────────────────────────────────────────────────────

import argparse
import math
import os
import sys
import time
import sqlite3
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.india.nse_client import NSEClient
from indicators import add_all_indicators
from probability.unified_cross_scorer import UnifiedCrossScorer
from probability.hmm_multiplexer      import HMMStrategyMultiplexer
from config.india_settings import INITIAL_CAPITAL_INR, INDIA_MAX_RISK_PER_TRADE_PCT
from engine.india_costs import calculate_round_trip_cost
from ml.meta_labeler import XGBoostMetaLabeler, add_hmm_regime_features

console = Console()
logger.add("logs/india_paper.log", rotation="10 MB", retention="30 days", level="INFO")

DB_FILE = "india_paper.sqlite"
IST = ZoneInfo("Asia/Kolkata")
MAX_POSITIONS_PER_MASTER = 3
MAX_OPEN_TRADES = 4

STOCK_ROUTING_MAP = {
    # ── Secular Mega-Cap Leaders -> NIFTY50 Macro Pullback Shield ──
    "INFY.NS":       "^NSEI",                 # IT
    "TCS.NS":        "^NSEI",                 # IT
    "LT.NS":         "^NSEI",                 # Infrastructure
    "BHARTIARTL.NS": "^NSEI",                 # Telecom
    "RELIANCE.NS":   "^NSEI",                 # Energy
    "NTPC.NS":       "^NSEI",                 # Energy
    "SUNPHARMA.NS":  "^NSEI",                 # Pharma
    "ITC.NS":        "^NSEI",                 # FMCG
}


@dataclass
class PaperPosition:
    symbol:       str
    master_index: str
    entry_time:   str
    entry_price:  float
    qty:          float
    sl:           float
    tp:           float
    sl_dist:      float


class SQLitePaperBroker:
    def __init__(self, initial_capital: float = INITIAL_CAPITAL_INR):
        self.initial_capital = initial_capital
        self.conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS state (
                    id INTEGER PRIMARY KEY,
                    cash REAL
                )
            """)
            res = self.conn.execute("SELECT cash FROM state WHERE id = 1").fetchone()
            if not res:
                self.conn.execute("INSERT INTO state (id, cash) VALUES (1, ?)", (self.initial_capital,))

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    master_index TEXT,
                    entry_time TEXT,
                    entry_price REAL,
                    qty REAL,
                    sl REAL,
                    tp REAL,
                    sl_dist REAL
                )
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    exit_type TEXT,
                    pnl REAL,
                    exit_time TEXT,
                    entry_time TEXT
                )
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS charts (
                    symbol TEXT PRIMARY KEY,
                    data_json TEXT
                )
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS india_ohlcv (
                    symbol TEXT,
                    interval TEXT,
                    timestamp TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, interval, timestamp)
                )
            """)

    @property
    def cash(self) -> float:
        return self.conn.execute("SELECT cash FROM state WHERE id = 1").fetchone()["cash"]

    @cash.setter
    def cash(self, value: float):
        with self.conn:
            self.conn.execute("UPDATE state SET cash = ? WHERE id = 1", (value,))

    def get_positions(self) -> dict[str, PaperPosition]:
        rows = self.conn.execute("SELECT * FROM positions").fetchall()
        pos_dict = {}
        for r in rows:
            pos_dict[r["symbol"]] = PaperPosition(
                symbol=r["symbol"], master_index=r["master_index"], entry_time=r["entry_time"],
                entry_price=r["entry_price"], qty=r["qty"], sl=r["sl"], tp=r["tp"], sl_dist=r["sl_dist"]
            )
        return pos_dict

    def add_position(self, p: PaperPosition):
        with self.conn:
            self.conn.execute("""
                INSERT INTO positions (symbol, master_index, entry_time, entry_price, qty, sl, tp, sl_dist)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (p.symbol, p.master_index, p.entry_time, p.entry_price, p.qty, p.sl, p.tp, p.sl_dist))

    def count_positions_on_master(self, master_index: str) -> int:
        positions = self.get_positions()
        return sum(1 for p in positions.values() if p.master_index == master_index)

    def update_position_sl(self, symbol: str, new_sl: float):
        with self.conn:
            self.conn.execute("UPDATE positions SET sl = ? WHERE symbol = ?", (new_sl, symbol))

    def remove_position(self, symbol: str):
        with self.conn:
            self.conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))

    def add_history(self, symbol: str, entry: float, exit_p: float, exit_type: str, pnl: float, exit_time: str, entry_time: str = None):
        with self.conn:
            self.conn.execute("""
                INSERT INTO history (symbol, entry_price, exit_price, exit_type, pnl, exit_time, entry_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, entry, exit_p, exit_type, pnl, exit_time, entry_time))

    def get_recent_history(self, symbol: str, limit: int = 10) -> list[dict]:
        rows = self.conn.execute("SELECT pnl, exit_type, exit_time FROM history WHERE symbol = ? ORDER BY exit_time DESC LIMIT ?", (symbol, limit)).fetchall()
        return [dict(r) for r in rows]

    def update_chart(self, symbol: str, df: pd.DataFrame):
        # Keep full dataset (typically ~250 daily candles = 1 year) for the dashboard UI
        tail_df = df.copy()

        # Format explicitly for tradingview lightweight charts
        chart_data = []
        for ts, r in tail_df.iterrows():
            time_str = ts.strftime('%Y-%m-%d')

            def safe_val(v):
                if pd.isna(v) or math.isnan(v): return None
                return round(float(v), 2)

            candle = {
                "time": time_str,
                "open": safe_val(r["open"]),
                "high": safe_val(r["high"]),
                "low": safe_val(r["low"]),
                "close": safe_val(r["close"]),
                "ema12": safe_val(r.get("ema_12")),
                "ema50": safe_val(r.get("ema_50")),
                "vwap": safe_val(r.get("vwap"))
            }
            chart_data.append(candle)

        import json
        json_str = json.dumps(chart_data)

        with self.conn:
            self.conn.execute("""
                INSERT INTO charts (symbol, data_json)
                VALUES (?, ?)
                ON CONFLICT(symbol) DO UPDATE SET data_json = excluded.data_json
            """, (symbol, json_str))

        self.save_ohlcv(symbol, "1d", df)

    def save_ohlcv(self, symbol: str, interval: str, df: pd.DataFrame):
        if df is None or df.empty:
            return
        records = []
        for ts, r in df.iterrows():
            if interval == "1d":
                time_str = ts.strftime('%Y-%m-%d')
            else:
                time_str = ts.strftime('%Y-%m-%d %H:%M:%S')
            records.append((
                symbol, interval, time_str,
                float(r.get("open", 0)), float(r.get("high", 0)),
                float(r.get("low", 0)), float(r.get("close", 0)),
                float(r.get("volume", 0))
            ))
        with self.conn:
            self.conn.executemany("""
                INSERT INTO india_ohlcv (symbol, interval, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, interval, timestamp) DO UPDATE SET
                    open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, volume=excluded.volume
            """, records)

    def print_portfolio(self, current_prices: dict[str, float]):
        console.print(f"\n[bold cyan]── PAPER PORTFOLIO STATUS (SQLite) ──[/]")
        console.print(f"  Available Cash: ₹{self.cash:,.2f}")

        positions = self.get_positions()
        if not positions:
            console.print("  [dim]No open positions.[/]\n")
            return

        tbl = Table(show_header=True, header_style="bold yellow")
        tbl.add_column("Symbol")
        tbl.add_column("Shares", justify="right")
        tbl.add_column("Entry", justify="right")
        tbl.add_column("LTP", justify="right")
        tbl.add_column("Unrealized P&L", justify="right")
        tbl.add_column("SL", justify="right")

        total_unrealized = 0.0
        for sym, p in positions.items():
            ltp = current_prices.get(sym, p.entry_price)
            gross = (ltp - p.entry_price) * p.qty
            total_unrealized += gross
            c_col = "green" if gross >= 0 else "red"
            tbl.add_row(
                sym, f"{p.qty:.1f}", f"₹{p.entry_price:,.2f}", f"₹{ltp:,.2f}",
                f"[{c_col}]₹{gross:,.2f}[/]", f"₹{p.sl:,.2f}"
            )
        console.print(tbl)
        console.print(f"  Net Unrealized: [{'green' if total_unrealized>=0 else 'red'}]₹{total_unrealized:,.2f}[/]\n")


def execute_market_scan(broker: SQLitePaperBroker):
    client = NSEClient()
    current_prices = {}
    candidate_entries = []

    console.print("\n[bold magenta]Running Master Index → Stock Routine...[/]")
    positions = broker.get_positions()

    try:
        df_nifty = add_all_indicators(client.get_ohlcv("^NSEI", "1d", 250))
        if df_nifty is not None and not df_nifty.empty:
            broker.update_chart("^NSEI", df_nifty)
            try:
                broker.save_ohlcv("^NSEI", "1h", client.get_ohlcv("^NSEI", "1h", 180))
                broker.save_ohlcv("^NSEI", "15m", client.get_ohlcv("^NSEI", "15m", 300))
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Failed to fetch NIFTY50 benchmark data for RS Gate: {e}")
        df_nifty = None

    # 1. Fetch data, manage existing positions, and collect candidate entries
    for stk, master in STOCK_ROUTING_MAP.items():
        try:
            logger.info(f"Analyzing {stk} via {master}...")
            try:
                df_idx = add_hmm_regime_features(add_all_indicators(client.get_ohlcv(master, "1d", 250)))
                df_stk = add_all_indicators(client.get_ohlcv(stk, "1d", 250))
            except Exception as net_e:
                logger.error(f"Network/Data failure on {stk}: {net_e}")
                continue

            if len(df_idx) < 5 or len(df_stk) < 5:
                continue

            last_idx_row = df_idx.iloc[-1]
            last_stk_row = df_stk.iloc[-1]
            ltp = float(last_stk_row["close"])
            current_prices[stk] = ltp

            # Serialize chart data for dashboard (Saves 1d to both charts and india_ohlcv)
            broker.update_chart(stk, df_stk)

            # Safely fetch intermediate/intraday bars for multi-timeframe dashboard charting
            try:
                df_1h = client.get_ohlcv(stk, "1h", 180)
                broker.save_ohlcv(stk, "1h", df_1h)
                df_15m = client.get_ohlcv(stk, "15m", 300)
                broker.save_ohlcv(stk, "15m", df_15m)
            except Exception as tf_e:
                logger.debug(f"Optional intraday TF fetch error for {stk}: {tf_e}")

            # Phase 1: Manage Existing Position
            if stk in positions:
                p = positions[stk]
                high, low = float(last_stk_row["high"]), float(last_stk_row["low"])
                exit_type = None
                exit_price = 0.0

                if low <= p.sl:
                    exit_type = "TRAIL_STOP" if p.sl >= p.entry_price else "STOP_LOSS"
                    exit_price = p.sl
                elif high >= p.tp:
                    exit_type = "TAKE_PROFIT"
                    exit_price = p.tp
                else:
                    # Trail SL
                    if high >= p.entry_price + p.sl_dist * 1.5 and p.sl < p.entry_price + p.sl_dist * 0.5:
                        new_sl = round(p.entry_price + p.sl_dist * 0.5, 2)
                        broker.update_position_sl(stk, new_sl)
                        logger.success(f"{stk} SL trailed to +0.5R lock-in (₹{new_sl})")
                    elif high >= p.entry_price + p.sl_dist * 1.0 and p.sl < p.entry_price:
                        new_sl = round(p.entry_price, 2)
                        broker.update_position_sl(stk, new_sl)
                        logger.success(f"{stk} SL trailed to Break-Even (₹{new_sl})")

                if exit_type:
                    gross = (exit_price - p.entry_price) * p.qty
                    b_c, s_c = calculate_round_trip_cost(p.entry_price, exit_price, p.qty, "CNC")
                    net = gross - (b_c.total + s_c.total)

                    broker.cash += (p.entry_price * p.qty) + net
                    broker.remove_position(stk)
                    broker.add_history(stk, p.entry_price, exit_price, exit_type, net, datetime.now(timezone.utc).isoformat(), p.entry_time)

                    logger.warning(f"CLOSED {stk} at ₹{exit_price} ({exit_type}) | PnL: ₹{net:.2f}")
                    continue

            # Phase 2: Collect Candidate Entry Signals (for non-position stocks)
            if stk not in positions and stk not in broker.get_positions():
                recent_trades = broker.get_recent_history(stk, limit=10)
                if len(recent_trades) >= 5:
                    wins = sum(1 for t in recent_trades if t["pnl"] > 0)
                    wr = (wins / len(recent_trades)) * 100.0
                    if wr < 40.0:
                        logger.warning(f"[QUARANTINE] {stk} rolling Win Rate is {wr:.1f}% (< 40% threshold over last {len(recent_trades)} trades). Skipping entry.")
                        continue

                if df_nifty is not None and len(df_nifty) >= 60 and len(df_stk) >= 60:
                    stk_close = float(df_stk.iloc[-1]["close"])
                    stk_close_60 = float(df_stk.iloc[-60]["close"])

                    nifty_close = float(df_nifty.iloc[-1]["close"])
                    nifty_close_60 = float(df_nifty.iloc[-60]["close"])

                    if nifty_close > 0 and nifty_close_60 > 0 and stk_close_60 > 0:
                        rs_today = stk_close / nifty_close
                        rs_60 = stk_close_60 / nifty_close_60
                        rs_slope = ((rs_today - rs_60) / rs_60) * 100.0

                        if rs_slope <= 0.0:
                            logger.info(f"[RS GATE] {stk} 60-day RS Slope vs NIFTY50 is {rs_slope:.2f}% (<= 0). Laggard blocked.")
                            continue

                scorer = HMMStrategyMultiplexer(symbol=master, interval="1d", atr_sl_mult=1.0, atr_tp_mult=4.0)
                signal = scorer.score(last_idx_row, df_idx)

                if signal.is_tradeable() and signal.direction == "LONG":
                    rs_today = ltp / float(last_idx_row["close"]) if float(last_idx_row["close"]) > 0 else 1.0
                    rs_60 = float(df_stk.iloc[-min(60, len(df_stk))]["close"]) / float(df_idx.iloc[-min(60, len(df_idx))]["close"]) if float(df_idx.iloc[-min(60, len(df_idx))]["close"]) > 0 else 1.0
                    rs_slope = ((rs_today - rs_60) / rs_60) * 100.0 if len(df_stk) >= 60 else 0.0
                    atr_stk = last_stk_row.get("atr", ltp * 0.015)
                    if math.isnan(atr_stk) or atr_stk <= 0: atr_stk = ltp * 0.015

                    feats = {
                        "hmm_state": float(last_idx_row.get("hmm_state", 0)),
                        "hmm_crash_score": float(last_idx_row.get("hmm_crash_score", 0.0)),
                        "rs_slope_60d": float(rs_slope),
                        "adx_14": float(last_idx_row.get("adx", 20.0)),
                        "vix": float(last_idx_row.get("vix", 16.0)),
                        "rsi_14": float(last_stk_row.get("rsi", 50.0)),
                        "atr_pct": float(atr_stk / ltp * 100.0) if ltp > 0 else 1.5,
                    }
                    candidate_entries.append((signal.probability, stk, master, ltp, last_stk_row, signal, feats))

        except Exception as e:
            logger.error(f"Global processing failure on {stk}: {e}")

    # 2. Prioritized Entry Execution (sorted descending by signal probability)
    candidate_entries.sort(key=lambda x: x[0], reverse=True)

    for prob, stk, master, ltp, last_stk_row, signal, feats in candidate_entries:
        try:
            total_open = len(broker.get_positions())
            if total_open >= MAX_OPEN_TRADES:
                logger.info(f"Global portfolio cap: {MAX_OPEN_TRADES} positions open. No new entries.")
                break

            if broker.count_positions_on_master(master) >= MAX_POSITIONS_PER_MASTER:
                logger.info(f"Concentration cap: {MAX_POSITIONS_PER_MASTER} position(s) already open on {master}. Skipping {stk} (score: {prob:.1f}).")
                continue

            p2 = float(feats.get("hmm_prob_2", 0.0))
            scalar = getattr(signal, "risk_amount", round(max(0.40, 1.0 - p2), 2))

            atr_stk = last_stk_row.get("atr")
            if math.isnan(atr_stk) or atr_stk <= 0: atr_stk = ltp * 0.015
            sl_dist = atr_stk * 1.0
            sl = round(ltp - sl_dist, 2)
            tp = round(ltp + sl_dist * 4.0, 2)

            cash = broker.cash
            current_equity = cash + sum(p.entry_price * p.qty for p in broker.get_positions().values())
            risk_inr = (current_equity * INDIA_MAX_RISK_PER_TRADE_PCT / 100.0) * scalar
            qty = risk_inr / sl_dist if sl_dist > 0 else 0
            max_affordable = cash / ltp if cash > 0 and ltp > 0 else 0
            qty = min(qty, max_affordable)

            if qty >= 1:
                cost = ltp * qty
                broker.cash -= cost

                pos = PaperPosition(
                    symbol=stk, master_index=master, entry_time=datetime.now(timezone.utc).isoformat(),
                    entry_price=ltp, qty=qty, sl=sl, tp=tp, sl_dist=sl_dist
                )
                broker.add_position(pos)
                logger.success(f"ENTERED {stk} LONG at ₹{ltp} | Qty: {qty:.2f} | Score: {prob:.1f} | Reason: {signal.reason}")

        except Exception as e:
            logger.error(f"Entry execution failure on {stk}: {e}")

    broker.print_portfolio(current_prices)


def run_daemon():
    """
    Production heartbeat.
    Wakes up, checks if NSE is open & it's time to trade (e.g. 15:15 IST).
    If triggered, executes scan. Then sleeps. Avoids tight polling.
    """
    logger.info("Initializing India Equity Paper Daemon...")
    broker = SQLitePaperBroker()

    # We want to run close to daily close to capture confirmed signals.
    # NSE closes at 15:30 IST. 15:15 is an institutional execution benchmark.
    TARGET_HOUR = 15
    TARGET_MIN = 15

    last_run_date = None

    while True:
        try:
            now_ist = datetime.now(IST)

            # Skip weekends (Saturday=5, Sunday=6)
            if now_ist.weekday() >= 5:
                time.sleep(3600)
                continue

            current_date = now_ist.date()

            # Trigger condition: Correct time and haven't run today
            if now_ist.hour == TARGET_HOUR and now_ist.minute >= TARGET_MIN:
                if last_run_date != current_date:
                    logger.info(f"Target execution window reached. Initiating {current_date} run.")
                    execute_market_scan(broker)
                    last_run_date = current_date

            # Smart Sleep
            # If past market hours, sleep until next morning
            if now_ist.hour >= 16:
                logger.debug("Market closed. Daemon sleeping until next morning.")
                time.sleep(28800)  # Sleep 8 hours
            else:
                # Polling frequency during market hours (checks time every 60s)
                time.sleep(60)

        except KeyboardInterrupt:
            logger.info("Daemon termination requested. Shutting down gracefully.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Daemon heartbeat fault: {e}")
            time.sleep(300) # Rest 5 min on catastrophic failure before retry


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--daemon", action="store_true", help="Start the forever-running background daemon")
    parser.add_argument("--max-per-master", type=int, default=3, help="Max open positions per master index")
    parser.add_argument("--max-open-trades", type=int, default=4, help="Max total open positions across entire book")
    args = parser.parse_args()

    MAX_POSITIONS_PER_MASTER = args.max_per_master
    MAX_OPEN_TRADES = args.max_open_trades

    try:
        import yfinance
    except ImportError:
        logger.warning("yfinance missing. Please install.")
        sys.exit(1)

    if args.daemon:
        run_daemon()
    else:
        # Run immediately once
        broker = SQLitePaperBroker()
        execute_market_scan(broker)
