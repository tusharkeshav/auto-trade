# ─────────────────────────────────────────────────────────────────
#  dashboard/web_server.py
#  TradingView-Base Institutional Quant Studio (v4.0)
#  Pure Python SPA • Full Indicator Suite • Volume Histogram • RSI(14)
#  Live Crosshair OHLCV HUD • Mansfield RS Badge • Fullscreen Studio
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, Any, List, Optional
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np

from engine.notifier import Notifier
from engine.paper_orchestrator_db import PaperOrchestratorDB
from config.india_settings import INDIA_UI_DEFAULT_VISIBLE_BARS

IST = ZoneInfo("Asia/Kolkata")
DB_PATH = ROOT_DIR / "ai_meta_paper_portfolio.sqlite"
TRACKED_SYMBOLS = [
    "GOLDBEES.NS", "^NSEI", "NIFTYBEES.NS", "BANKBEES.NS", "CPSEETF.NS",
    "TCS.NS", "INFY.NS", "RELIANCE.NS", "LT.NS", "BHARTIARTL.NS", "SBIN.NS"
]


# ─────────────────────────────────────────────────────────────────
#  Backend State Provider & Technical Data Calculations
# ─────────────────────────────────────────────────────────────────

def get_live_portfolio_data() -> Dict[str, Any]:
    """Reads live portfolio state, open positions, and trade logs from SQLite."""
    res = {
        "cash": 100000.0,
        "invested": 0.0,
        "total_nav": 100000.0,
        "peak_nav": 100000.0,
        "realized_pnl": 0.0,
        "total_taxes": 0.0,
        "active_regime": "BEAR_DEFENSE",
        "last_updated": "—",
        "positions": [],
        "trades": [],
        "nav_history": [],
        "telegram_users": ["Infinity (1292507208)", "Nikhil Keshav (8931985247)"],
        "cron_status": "Active (15m Polling & @reboot)"
    }

    if DB_PATH.exists():
        try:
            with sqlite3.connect(str(DB_PATH), uri=True) as conn:
                conn.row_factory = sqlite3.Row
                p = conn.execute("SELECT * FROM portfolio_state WHERE id=1").fetchone()
                if p:
                    res["cash"] = round(p["cash_balance_inr"], 2)
                    res["invested"] = round(p["invested_inr"], 2)
                    res["total_nav"] = round(p["current_nav_inr"], 2)
                    res["peak_nav"] = round(p["peak_nav_inr"], 2)
                    res["realized_pnl"] = round(p["realized_pnl_inr"], 2)
                    res["total_taxes"] = round(p["total_taxes_paid_inr"], 2)
                    res["active_regime"] = p["active_regime"]
                    res["last_updated"] = p["updated_at"]

                pos_rows = conn.execute("SELECT * FROM positions").fetchall()
                for r in pos_rows:
                    entry = r["entry_price"]
                    curr = r["current_price"]
                    qty = r["quantity"]
                    pnl = (curr - entry) * qty
                    pnl_pct = ((curr - entry) / entry * 100.0) if entry > 0 else 0.0
                    res["positions"].append({
                        "symbol": r["symbol"],
                        "strategy": r["strategy_name"],
                        "entry_date": r["entry_date"][:10],
                        "entry": round(entry, 2),
                        "current": round(curr, 2),
                        "qty": qty,
                        "stop_loss": round(r["stop_loss"], 2),
                        "take_profit": round(r["take_profit"], 2),
                        "unrealized_pnl": round(pnl, 2),
                        "unrealized_pnl_pct": round(pnl_pct, 2),
                        "be_locked": bool(r["be_locked"]),
                    })

                tr_rows = conn.execute("SELECT * FROM trade_history ORDER BY trade_id DESC LIMIT 50").fetchall()
                for r in tr_rows:
                    res["trades"].append({
                        "symbol": r["symbol"],
                        "strategy": r["strategy_name"],
                        "entry_date": r["entry_date"][:10],
                        "exit_date": r["exit_date"][:10],
                        "entry_price": round(r["entry_price"], 2),
                        "exit_price": round(r["exit_price"], 2),
                        "qty": r["quantity"],
                        "gross_pnl": round(r["gross_pnl"], 2),
                        "taxes": round(r["taxes_inr"], 2),
                        "net_pnl": round(r["net_pnl"], 2),
                        "net_pnl_pct": round(r["net_pnl_pct"], 2),
                        "reason": r["exit_reason"],
                        "bars_held": r["bars_held"],
                    })

                nav_rows = conn.execute("SELECT * FROM nav_history ORDER BY date ASC").fetchall()
                for r in nav_rows:
                    res["nav_history"].append({
                        "date": r["date"],
                        "nav": round(r["nav_inr"], 2),
                        "cash": round(r["cash_inr"], 2),
                        "invested": round(r["invested_inr"], 2),
                        "regime": r["regime"]
                    })
        except Exception as e:
            print(f"[Web Dashboard] SQLite read error: {e}")

    return res


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def fetch_candles_for_chart(symbol: str, interval: str = "1d") -> Dict[str, Any]:
    """Fetches OHLCV + Volume + 200 SMA, 50 EMA, 20 SMA, 12 EMA, VWAP, RSI(14) and Mansfield RS."""
    try:
        import yfinance as yf
        valid_sym = symbol if symbol.endswith(".NS") or symbol.startswith("^") else f"{symbol}.NS"
        download_syms = [valid_sym, "^NSEI"] if valid_sym != "^NSEI" else ["^NSEI"]
        df_raw = yf.download(download_syms, period="2y" if interval == "1d" else "1mo", interval=interval, progress=False)
        if df_raw is None or df_raw.empty:
            return {"candles": [], "mansfield_rs": 0.0}

        if isinstance(df_raw.columns, pd.MultiIndex):
            cols = {}
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                if (c, valid_sym) in df_raw.columns:
                    cols[c] = df_raw[(c, valid_sym)]
                elif c in df_raw.columns:
                    cols[c] = df_raw[c][valid_sym] if valid_sym in df_raw[c] else df_raw[c].iloc[:, 0]
            sub_df = pd.DataFrame(cols, index=df_raw.index).ffill().dropna()

            # NIFTY close for Mansfield RS
            if ("Close", "^NSEI") in df_raw.columns:
                nifty_close = df_raw[("Close", "^NSEI")].ffill().dropna()
            elif "Close" in df_raw.columns and "^NSEI" in df_raw["Close"]:
                nifty_close = df_raw["Close"]["^NSEI"].ffill().dropna()
            else:
                nifty_close = sub_df["Close"]
        else:
            sub_df = df_raw.ffill().dropna()
            nifty_close = sub_df["Close"]

        # Compute Technical Overlays
        close_s = sub_df["Close"]
        sub_df["ema12"] = close_s.ewm(span=12, adjust=False).mean()
        sub_df["sma20"] = close_s.rolling(20).mean()
        sub_df["ema50"] = close_s.ewm(span=50, adjust=False).mean()
        sub_df["sma200"] = close_s.rolling(200).mean()
        sub_df["rsi14"] = calculate_rsi(close_s, 14)
        sub_df["vol_ma20"] = sub_df["Volume"].rolling(20).mean()

        # VWAP
        typical = (sub_df["High"] + sub_df["Low"] + sub_df["Close"]) / 3.0
        cum_vol = sub_df["Volume"].cumsum()
        cum_vp = (typical * sub_df["Volume"]).cumsum()
        sub_df["vwap"] = cum_vp / cum_vol.replace(0, 1)

        # Mansfield RS vs NIFTY (60d change)
        mansfield_score = 0.0
        try:
            if len(sub_df) >= 60 and len(nifty_close) >= 60:
                rs_now = float(close_s.iloc[-1]) / float(nifty_close.iloc[-1])
                rs_60 = float(close_s.iloc[-60]) / float(nifty_close.iloc[-60])
                mansfield_score = round(((rs_now - rs_60) / rs_60) * 100.0, 2)
        except Exception:
            pass

        candles = []
        for idx, row in sub_df.iterrows():
            d_str = idx.strftime("%Y-%m-%d") if interval == "1d" else int(idx.timestamp())
            is_green = float(row["Close"]) >= float(row["Open"])
            candles.append({
                "time": d_str,
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row.get("Volume", 0)),
                "vol_color": "rgba(16, 185, 129, 0.4)" if is_green else "rgba(244, 63, 94, 0.4)",
                "vol_ma20": round(float(row["vol_ma20"]), 2) if pd.notnull(row["vol_ma20"]) else None,
                "ema12": round(float(row["ema12"]), 2) if pd.notnull(row["ema12"]) else None,
                "sma20": round(float(row["sma20"]), 2) if pd.notnull(row["sma20"]) else None,
                "ema50": round(float(row["ema50"]), 2) if pd.notnull(row["ema50"]) else None,
                "sma200": round(float(row["sma200"]), 2) if pd.notnull(row["sma200"]) else None,
                "vwap": round(float(row["vwap"]), 2) if pd.notnull(row["vwap"]) else None,
                "rsi14": round(float(row["rsi14"]), 2) if pd.notnull(row["rsi14"]) else 50.0,
            })

        return {"candles": candles, "mansfield_rs": mansfield_score}
    except Exception as e:
        print(f"[Chart Fetcher] Error fetching {symbol}: {e}")
        return {"candles": [], "mansfield_rs": 0.0}


# ─────────────────────────────────────────────────────────────────
#  Self-Contained Glassmorphic Single Page Application HTML/JS
# ─────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Quant Meta-Orchestrator • TradingView Studio v4.0</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://unpkg.com/lightweight-charts@3.8.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root {
    --bg: #060911;
    --surface: #0c1220;
    --surface-hover: #121b2f;
    --card-border: rgba(255, 255, 255, 0.07);
    --accent: #3b82f6;
    --accent-glow: rgba(59, 130, 246, 0.4);
    --green: #10b981;
    --green-glow: rgba(16, 185, 129, 0.25);
    --red: #f43f5e;
    --yellow: #f59e0b;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text-primary);
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 13px;
    line-height: 1.5;
    overflow-x: hidden;
  }
  .mono { font-family: 'JetBrains Mono', monospace; }

  /* Top Navigation Ribbon */
  header {
    background: rgba(12, 18, 32, 0.88);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--card-border);
    padding: 12px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
    gap: 16px;
    flex-wrap: wrap;
  }
  .brand-block { display: flex; align-items: center; gap: 12px; }
  .brand-logo {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    box-shadow: 0 0 16px var(--accent-glow);
  }
  .brand-title { font-size: 16px; font-weight: 800; letter-spacing: -0.4px; color: #fff; }
  .badge-env {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.35);
    color: var(--yellow);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 700;
  }

  .header-metrics { display: flex; align-items: center; gap: 12px; }
  .pill {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--card-border);
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    display: flex; align-items: center; gap: 6px;
  }
  .pill-regime {
    background: rgba(244, 63, 94, 0.12);
    border-color: rgba(244, 63, 94, 0.3);
    color: #fda4af;
    font-weight: 700;
  }
  .pill-regime.bull {
    background: rgba(16, 185, 129, 0.12);
    border-color: rgba(16, 185, 129, 0.3);
    color: #6ee7b7;
  }

  .action-btn {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    color: #fff;
    border: 1px solid rgba(255,255,255,0.15);
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    display: flex; align-items: center; gap: 6px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
  }
  .action-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.45);
  }
  .action-btn-ghost {
    background: rgba(255, 255, 255, 0.05);
    color: var(--text-secondary);
    border: 1px solid var(--card-border);
    box-shadow: none;
  }
  .action-btn-ghost:hover {
    background: rgba(255, 255, 255, 0.09);
    color: #fff;
  }

  /* Main Container */
  .container { max-width: 1680px; margin: 0 auto; padding: 20px 28px; }

  /* 4-Card Hero Financial HUD */
  .hud-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 20px;
  }
  .hud-card {
    background: var(--surface);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 16px 20px;
    position: relative;
    overflow: hidden;
    transition: all 0.2s ease;
  }
  .hud-card:hover {
    border-color: rgba(255, 255, 255, 0.15);
    transform: translateY(-2px);
  }
  .hud-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 100%; height: 3px;
    background: var(--accent);
  }
  .hud-card.green::before { background: var(--green); }
  .hud-card.yellow::before { background: var(--yellow); }
  .hud-card.red::before { background: var(--red); }

  .hud-label {
    font-size: 11px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
  }
  .hud-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    color: #fff;
    margin-bottom: 4px;
  }
  .hud-sub {
    font-size: 11px;
    color: var(--text-secondary);
    display: flex; align-items: center; gap: 6px;
  }

  /* 2-Column Pro Trader Studio */
  .studio-grid {
    display: grid;
    grid-template-columns: 1.45fr 1fr;
    gap: 20px;
    margin-bottom: 20px;
  }
  @media (max-width: 1100px) {
    .studio-grid { grid-template-columns: 1fr; }
    .hud-grid { grid-template-columns: 1fr 1fr; }
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 20px;
    position: relative;
  }
  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--card-border);
    flex-wrap: wrap;
    gap: 10px;
  }
  .card-title {
    font-size: 12px;
    font-weight: 800;
    color: var(--text-secondary);
    letter-spacing: 0.8px;
    text-transform: uppercase;
    display: flex; align-items: center; gap: 8px;
  }

  /* Asset Pills Selector */
  .asset-pills { display: flex; gap: 6px; flex-wrap: wrap; }
  .asset-pill {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--card-border);
    color: var(--text-secondary);
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .asset-pill:hover { background: rgba(255, 255, 255, 0.09); color: #fff; }
  .asset-pill.active {
    background: rgba(59, 130, 246, 0.18);
    border-color: var(--accent);
    color: #60a5fa;
    box-shadow: 0 0 10px rgba(59, 130, 246, 0.25);
  }

  /* Live Crosshair OHLCV Tooltip HUD */
  .tv-hud-bar {
    display: flex;
    align-items: center;
    gap: 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-secondary);
    background: rgba(0, 0, 0, 0.25);
    padding: 6px 12px;
    border-radius: 6px;
    margin-bottom: 10px;
    border: 1px solid rgba(255, 255, 255, 0.04);
    overflow-x: auto;
    white-space: nowrap;
  }
  .tv-hud-bar span b { color: #fff; }

  /* Technical Indicator Toggles Bar */
  .ind-pills-bar {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 12px;
    padding: 8px 12px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--card-border);
    border-radius: 8px;
  }
  .ind-btn {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--card-border);
    color: var(--text-muted);
    padding: 3px 9px;
    border-radius: 5px;
    font-size: 11px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    cursor: pointer;
    transition: all 0.15s;
  }
  .ind-btn:hover { color: #fff; border-color: rgba(255,255,255,0.2); }
  .ind-btn.active-ema12  { background: rgba(96, 165, 250, 0.15); border-color: #60a5fa; color: #93c5fd; }
  .ind-btn.active-sma20  { background: rgba(192, 132, 252, 0.15); border-color: #c084fc; color: #e9d5ff; }
  .ind-btn.active-ema50  { background: rgba(251, 146, 60, 0.15); border-color: #fb923c; color: #fed7aa; }
  .ind-btn.active-sma200 { background: rgba(245, 158, 11, 0.2); border-color: #f59e0b; color: #fde68a; font-weight: 800; }
  .ind-btn.active-vwap   { background: rgba(226, 232, 240, 0.15); border-color: #e2e8f0; color: #f8fafc; }
  .ind-btn.active-rsi    { background: rgba(168, 85, 247, 0.15); border-color: #a855f7; color: #d8b4fe; }
  .ind-btn.active-sltp   { background: rgba(16, 185, 129, 0.15); border-color: #10b981; color: #6ee7b7; }

  /* Box Zoom Overlay */
  #box-zoom-overlay {
    position: absolute;
    background: rgba(59, 130, 246, 0.15);
    border: 1px dashed #3b82f6;
    pointer-events: none;
    display: none;
    z-index: 50;
  }

  /* Visual Risk Distance Meter */
  .risk-gauge-box {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
  }
  .risk-meter-track {
    height: 8px;
    background: #1e293b;
    border-radius: 4px;
    position: relative;
    margin: 14px 0 8px 0;
  }
  .risk-meter-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #f43f5e 0%, #f59e0b 50%, #10b981 100%);
  }
  .risk-meter-thumb {
    position: absolute;
    top: -4px;
    width: 16px;
    height: 16px;
    background: #fff;
    border: 2px solid var(--accent);
    border-radius: 50%;
    transform: translateX(-50%);
    box-shadow: 0 0 10px #3b82f6;
  }
  .risk-meter-labels {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-secondary);
  }

  /* Table styling */
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th {
    color: var(--text-muted);
    font-weight: 700;
    font-size: 10px;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid var(--card-border);
  }
  td {
    padding: 11px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    vertical-align: middle;
  }
  tr:hover td { background: rgba(255,255,255,0.02); }

  .pos { color: var(--green); font-weight: 600; }
  .neg { color: var(--red); font-weight: 600; }

  /* Bottom Tabbed Drawer */
  .tab-nav { display: flex; gap: 8px; border-bottom: 1px solid var(--card-border); margin-bottom: 16px; }
  .tab-btn {
    background: transparent;
    border: none;
    color: var(--text-muted);
    padding: 8px 16px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    position: relative;
    transition: all 0.2s;
  }
  .tab-btn:hover { color: #fff; }
  .tab-btn.active { color: var(--accent); }
  .tab-btn.active::after {
    content: '';
    position: absolute;
    bottom: -1px; left: 0; width: 100%; height: 2px;
    background: var(--accent);
  }

  /* Fullscreen Mode */
  .fullscreen-card {
    position: fixed !important;
    top: 0 !important; left: 0 !important;
    width: 100vw !important; height: 100vh !important;
    z-index: 9999 !important;
    border-radius: 0 !important;
    background: var(--bg) !important;
    padding: 24px !important;
    overflow: auto;
  }

  /* Toast Notification */
  #toast {
    position: fixed;
    bottom: 24px; right: 28px;
    background: #1e293b;
    color: #fff;
    border: 1px solid #3b82f6;
    border-radius: 8px;
    padding: 12px 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.8);
    display: none;
    z-index: 1000;
    font-weight: 600;
    animation: fadeIn 0.3s ease;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
</style>
</head>
<body>

<header>
  <div class="brand-block">
    <div class="brand-logo">📈</div>
    <div>
      <div class="brand-title">QUANT META-ORCHESTRATOR</div>
      <div style="font-size: 10px; color: var(--text-muted);">TradingView Studio v4.0 • Zero External Deps</div>
    </div>
    <div class="badge-env">CNC CASH • NSE INDIA</div>
  </div>

  <div class="header-metrics">
    <div id="regime-badge" class="pill pill-regime">🛡️ BEAR_DEFENSE</div>
    <div id="session-pill" class="pill">⏰ Checking Market Session...</div>
    <div id="telegram-pill" class="pill" style="color:#60a5fa;">📱 Telegram 2x Active</div>
  </div>

  <div style="display: flex; gap: 8px;">
    <button class="action-btn-ghost action-btn" onclick="triggerTestNotify()">📨 Test Telegram</button>
    <button class="action-btn" onclick="triggerMarketScan()">⚡ Scan Market Now</button>
  </div>
</header>

<div class="container">

  <!-- 4-Card Hero Financial HUD -->
  <div class="hud-grid">
    <div class="hud-card green">
      <div class="hud-label">Total Portfolio NAV</div>
      <div class="hud-value" id="nav-val">₹1,00,000.00</div>
      <div class="hud-sub"><span class="pos" id="nav-gain">+0.0%</span> &nbsp;• Peak DD: <span id="nav-dd">0.0%</span></div>
    </div>
    <div class="hud-card yellow">
      <div class="hud-label">Available Cash Reserve</div>
      <div class="hud-value" id="cash-val">₹1,00,000.00</div>
      <div class="hud-sub" id="cash-sub">Liquid Buying Power</div>
    </div>
    <div class="hud-card green">
      <div class="hud-label">Floating Unrealized P&L</div>
      <div class="hud-value" id="unreal-val">+₹0.00</div>
      <div class="hud-sub"><span class="pos" id="unreal-pct">+0.0%</span> on active holdings</div>
    </div>
    <div class="hud-card">
      <div class="hud-label">Macro 200 SMA Shield</div>
      <div class="hud-value" id="shield-val">BEAR DEFENSE</div>
      <div class="hud-sub" id="shield-sub">100% Capital in Gold Shield</div>
    </div>
  </div>

  <!-- 2-Column Pro Trader Studio -->
  <div class="studio-grid">

    <!-- Left (60%): Interactive TradingView Pro Chart Studio -->
    <div class="card" id="chart-card">
      <div class="card-header">
        <div class="asset-pills" id="asset-pills">
          <button class="asset-pill active" onclick="loadChart('GOLDBEES.NS')">🪙 GOLDBEES</button>
          <button class="asset-pill" onclick="loadChart('^NSEI')">📊 NIFTY 50</button>
          <button class="asset-pill" onclick="loadChart('TCS.NS')">TCS</button>
          <button class="asset-pill" onclick="loadChart('INFY.NS')">INFY</button>
          <button class="asset-pill" onclick="loadChart('RELIANCE.NS')">RELIANCE</button>
          <button class="asset-pill" onclick="loadChart('LT.NS')">LT</button>
          <button class="asset-pill" onclick="loadChart('BANKBEES.NS')">BANKBEES</button>
          <button class="asset-pill" onclick="loadChart('CPSEETF.NS')">CPSEETF</button>
        </div>

        <div style="display:flex; gap:6px; align-items:center;">
          <button class="asset-pill" id="tf-1d" onclick="setTimeframe('1d')">1D</button>
          <button class="asset-pill" id="tf-1h" onclick="setTimeframe('1h')">1H</button>
          <button class="asset-pill" id="tf-15m" onclick="setTimeframe('15m')">15M</button>
          <button class="action-btn-ghost action-btn" id="btn-collapse" onclick="toggleCollapseChart()" style="padding:3px 8px; font-size:11px;" title="Minimize/Collapse Chart">─</button>
          <button class="action-btn-ghost action-btn" id="btn-fs" onclick="toggleFullscreen()" style="padding:3px 8px; font-size:11px;" title="Maximize/Minimize Fullscreen (ESC)">⛶ Maximize</button>
        </div>
      </div>

      <div id="chart-body">
      <!-- Live Crosshair OHLCV Tooltip HUD -->
      <div class="tv-hud-bar" id="tv-hud">
        <span>O: <b id="hud-o">—</b></span>
        <span>H: <b id="hud-h">—</b></span>
        <span>L: <b id="hud-l">—</b></span>
        <span>C: <b id="hud-c">—</b></span>
        <span>VOL: <b id="hud-v">—</b></span>
        <span>EMA12: <b id="hud-e12" style="color:#60a5fa;">—</b></span>
        <span>SMA200: <b id="hud-s200" style="color:#f59e0b;">—</b></span>
        <span>RSI(14): <b id="hud-rsi" style="color:#c084fc;">—</b></span>
        <span id="hud-rs-badge" class="badge-env" style="margin-left:auto;">RS vs NIFTY: +0.0%</span>
      </div>

      <!-- Full Technical Indicators & Zoom Tools Bar -->
      <div class="ind-pills-bar">
        <button id="btn-ind-ema12" class="ind-btn active-ema12" onclick="toggleInd('ema12')">EMA 12</button>
        <button id="btn-ind-sma20" class="ind-btn active-sma20" onclick="toggleInd('sma20')">SMA 20</button>
        <button id="btn-ind-ema50" class="ind-btn active-ema50" onclick="toggleInd('ema50')">EMA 50</button>
        <button id="btn-ind-sma200" class="ind-btn active-sma200" onclick="toggleInd('sma200')">SMA 200 🛡️</button>
        <button id="btn-ind-vwap" class="ind-btn active-vwap" onclick="toggleInd('vwap')">VWAP</button>
        <button id="btn-ind-rsi" class="ind-btn active-rsi" onclick="toggleInd('rsi')">RSI(14)</button>
        <button id="btn-ind-sltp" class="ind-btn active-sltp" onclick="toggleInd('sltp')">📍 SL/TP Lines</button>
        <span style="color:var(--card-border); margin: 0 4px;">|</span>
        <select id="chart-zoom-select" onchange="applyZoomRange(this.value)" style="background:#1e293b; color:#e2e8f0; font-weight:600; border:1px solid var(--card-border); border-radius:5px; padding:3px 8px; font-family:'JetBrains Mono',monospace; font-size:11px;">
           <option value="default">Zoom: Default</option>
           <option value="1w">Zoom: 1 Week</option>
           <option value="1m">Zoom: 1 Month</option>
           <option value="3m">Zoom: 3 Months</option>
           <option value="6m">Zoom: 6 Months</option>
           <option value="1y">Zoom: 1 Year</option>
           <option value="all">Zoom: ALL</option>
        </select>
      </div>

      <div style="position: relative; width: 100%;">
        <div id="chart-container" style="width: 100%; height: 380px;"></div>
        <div id="rsi-container" style="width: 100%; height: 120px; border-top: 1px solid var(--card-border); margin-top: 6px;"></div>
        <div id="box-zoom-overlay"></div>
      </div>
      </div> <!-- /#chart-body -->
    </div>

    <!-- Right (40%): Live Position & Risk Inspector -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">🎯 ACTIVE POSITION & RISK GAUGES</div>
        <div class="pill" id="holding-count" style="font-size:10px;">1 Holding Active</div>
      </div>

      <div id="position-inspector-content">
        <!-- Rendered Dynamically by JS -->
      </div>
    </div>

  </div>

  <!-- Bottom Drawer: History & Broadcast Logs -->
  <div class="card">
    <div class="tab-nav">
      <button class="tab-btn active" id="tab-tr" onclick="switchBottomTab('trades')">📋 Audited Trade History Ledger</button>
      <button class="tab-btn" id="tab-tg" onclick="switchBottomTab('telegram')">📱 Telegram & Crontab Health</button>
    </div>

    <div id="bottom-content">
      <!-- Dynamic Table Rendered by JS -->
    </div>
  </div>

</div>

<div id="toast"></div>

<script>
  const $ = id => document.getElementById(id);
  const fmt = n => Number(n || 0).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

  let activeSymbol = "GOLDBEES.NS";
  let activeInterval = "1d";
  let tvChart = null;
  let rsiChart = null;
  let candleSeries = null, volumeSeries = null;
  let ema12Series = null, sma20Series = null, ema50Series = null, sma200Series = null, vwapSeries = null;
  let rsiSeries = null, rsi70Line = null, rsi30Line = null, rsi50Line = null;
  let priceLines = [];
  let currentPortfolioData = null;
  let currentCandleData = [];
  let activeBottomTab = "trades";
  let isFullscreen = false;

  let visEma12 = true, visSma20 = true, visEma50 = true, visSma200 = true, visVwap = true, visRsi = true, visSLTP = true;

  // Toast notifications
  function showToast(msg) {
    const t = $('toast');
    t.innerHTML = msg;
    t.style.display = 'block';
    setTimeout(() => { t.style.display = 'none'; }, 4000);
  }

  // Initialize TradingView Main Chart & RSI Sub-Chart
  function initChart() {
    const container = $('chart-container');
    container.innerHTML = '';
    tvChart = LightweightCharts.createChart(container, {
      width: container.clientWidth,
      height: 380,
      layout: { backgroundColor: '#0c1220', textColor: '#94a3b8' },
      grid: { vertLines: { color: 'rgba(255,255,255,0.03)' }, horzLines: { color: 'rgba(255,255,255,0.03)' } },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      priceScale: { borderColor: 'rgba(255,255,255,0.08)', scaleMargins: { top: 0.1, bottom: 0.2 } },
      timeScale: { borderColor: 'rgba(255,255,255,0.08)', timeVisible: true, rightOffset: 5, barSpacing: 9 }
    });

    candleSeries = tvChart.addCandlestickSeries({
      upColor: '#10b981', downColor: '#f43f5e',
      borderUpColor: '#10b981', borderDownColor: '#f43f5e',
      wickUpColor: '#10b981', wickDownColor: '#f43f5e'
    });

    // Volume Histogram (Lower sub-pane)
    volumeSeries = tvChart.addHistogramSeries({
      color: 'rgba(16, 185, 129, 0.4)',
      priceFormat: { type: 'volume' },
      priceScaleId: '', // Overlay scale
      scaleMargins: { top: 0.82, bottom: 0 }
    });

    // Technical Indicator Line Series
    ema12Series  = tvChart.addLineSeries({ color: '#60a5fa', lineWidth: 1.5, title: 'EMA12' });
    sma20Series  = tvChart.addLineSeries({ color: '#c084fc', lineWidth: 1.5, title: 'SMA20' });
    ema50Series  = tvChart.addLineSeries({ color: '#fb923c', lineWidth: 1.5, title: 'EMA50' });
    sma200Series = tvChart.addLineSeries({ color: '#f59e0b', lineWidth: 2.2, title: 'SMA200 🛡️' });
    vwapSeries   = tvChart.addLineSeries({ color: '#e2e8f0', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, title: 'VWAP' });

    // RSI Sub-Chart Canvas
    const rsiCont = $('rsi-container');
    rsiCont.innerHTML = '';
    rsiChart = LightweightCharts.createChart(rsiCont, {
      width: rsiCont.clientWidth,
      height: 120,
      layout: { backgroundColor: '#0c1220', textColor: '#94a3b8' },
      grid: { vertLines: { color: 'rgba(255,255,255,0.03)' }, horzLines: { color: 'rgba(255,255,255,0.03)' } },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      priceScale: { borderColor: 'rgba(255,255,255,0.08)', scaleMargins: { top: 0.1, bottom: 0.1 } },
      timeScale: { borderColor: 'rgba(255,255,255,0.08)', timeVisible: true, rightOffset: 5, barSpacing: 9 }
    });

    rsiSeries = rsiChart.addLineSeries({ color: '#c084fc', lineWidth: 1.8, title: 'RSI(14)' });
    rsi70Line = rsiChart.addLineSeries({ color: 'rgba(244, 63, 94, 0.4)', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed });
    rsi50Line = rsiChart.addLineSeries({ color: 'rgba(255, 255, 255, 0.2)', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted });
    rsi30Line = rsiChart.addLineSeries({ color: 'rgba(16, 185, 129, 0.4)', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed });

    // Synchronize crosshairs between Main Chart & RSI
    tvChart.subscribeCrosshairMove(param => {
      if (!param || !param.time || !param.seriesPrices) return;
      const bar = param.seriesPrices.get(candleSeries);
      if (bar) {
        $('hud-o').textContent = `₹${fmt(bar.open)}`;
        $('hud-h').textContent = `₹${fmt(bar.high)}`;
        $('hud-l').textContent = `₹${fmt(bar.low)}`;
        $('hud-c').textContent = `₹${fmt(bar.close)}`;
      }
      const e12 = param.seriesPrices.get(ema12Series);
      if (e12) $('hud-e12').textContent = `₹${fmt(e12)}`;
      const s200 = param.seriesPrices.get(sma200Series);
      if (s200) $('hud-s200').textContent = `₹${fmt(s200)}`;
      const vol = param.seriesPrices.get(volumeSeries);
      if (vol) $('hud-v').textContent = `${(vol / 1000000).toFixed(2)}M`;
    });

    // Synchronize visible ranges between Main & RSI charts
    let isSyncingRange = false;
    tvChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
      if (isSyncingRange || !range || !rsiChart) return;
      isSyncingRange = true;
      rsiChart.timeScale().setVisibleLogicalRange(range);
      isSyncingRange = false;
    });
    rsiChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
      if (isSyncingRange || !range || !tvChart) return;
      isSyncingRange = true;
      tvChart.timeScale().setVisibleLogicalRange(range);
      isSyncingRange = false;
    });

    initBoxZoom();
    loadChart(activeSymbol);
  }

  async function loadChart(sym) {
    activeSymbol = sym;
    document.querySelectorAll('.asset-pill').forEach(b => {
      if (b.textContent.includes(sym.replace('.NS', '').replace('^', ''))) b.classList.add('active');
      else b.classList.remove('active');
    });

    try {
      const res = await fetch(`/api/candles?symbol=${sym}&interval=${activeInterval}`);
      if (!res.ok) return;
      const payload = await res.json();
      const data = payload.candles || [];
      if (!data || data.length === 0) return;
      currentCandleData = data;

      // Update Mansfield RS Badge
      const rs = payload.mansfield_rs || 0.0;
      $('hud-rs-badge').textContent = `60d RS vs NIFTY: ${rs >= 0 ? '+' : ''}${rs}% ${rs >= 0 ? '🟢' : '🔴'}`;
      $('hud-rs-badge').className = `badge-env ${rs >= 0 ? 'pos' : 'neg'}`;

      candleSeries.setData(data.map(d => ({
        time: d.time, open: d.open, high: d.high, low: d.low, close: d.close
      })));

      volumeSeries.setData(data.map(d => ({
        time: d.time, value: d.volume, color: d.vol_color
      })));

      // Indicator datasets
      ema12Series.setData(data.filter(d => d.ema12 != null).map(d => ({ time: d.time, value: d.ema12 })));
      sma20Series.setData(data.filter(d => d.sma20 != null).map(d => ({ time: d.time, value: d.sma20 })));
      ema50Series.setData(data.filter(d => d.ema50 != null).map(d => ({ time: d.time, value: d.ema50 })));
      sma200Series.setData(data.filter(d => d.sma200 != null).map(d => ({ time: d.time, value: d.sma200 })));
      vwapSeries.setData(data.filter(d => d.vwap != null).map(d => ({ time: d.time, value: d.vwap })));

      // RSI datasets
      rsiSeries.setData(data.map(d => ({ time: d.time, value: d.rsi14 })));
      rsi70Line.setData(data.map(d => ({ time: d.time, value: 70 })));
      rsi50Line.setData(data.map(d => ({ time: d.time, value: 50 })));
      rsi30Line.setData(data.map(d => ({ time: d.time, value: 30 })));

      applyIndicatorVisibility();
      renderPriceLines(sym);

      // Smart default zoom: Show last 90 trading bars so candles are thick and fill the canvas
      const totalBars = data.length;
      if (totalBars > 80) {
        tvChart.timeScale().setVisibleLogicalRange({ from: totalBars - 90, to: totalBars + 2 });
        rsiChart.timeScale().setVisibleLogicalRange({ from: totalBars - 90, to: totalBars + 2 });
      } else {
        tvChart.timeScale().fitContent();
        rsiChart.timeScale().fitContent();
      }
    } catch(e) {}
  }

  function renderPriceLines(sym) {
    priceLines.forEach(pl => { try { candleSeries.removePriceLine(pl); } catch(e){} });
    priceLines = [];

    if (!visSLTP) return;

    if (currentPortfolioData && currentPortfolioData.positions) {
      const pos = currentPortfolioData.positions.find(p => p.symbol === sym);
      if (pos) {
        priceLines.push(candleSeries.createPriceLine({
          price: pos.entry, color: '#3b82f6', lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Solid, title: `ENTRY @ ₹${pos.entry}`
        }));
        priceLines.push(candleSeries.createPriceLine({
          price: pos.stop_loss, color: '#f43f5e', lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Dashed, title: `SL @ ₹${pos.stop_loss}`
        }));
        priceLines.push(candleSeries.createPriceLine({
          price: pos.take_profit, color: '#10b981', lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Dashed, title: `TARGET @ ₹${pos.take_profit}`
        }));
      }
    }
  }

  function toggleInd(ind) {
    if (ind === 'ema12')  { visEma12 = !visEma12;   $('btn-ind-ema12').classList.toggle('active-ema12', visEma12); }
    if (ind === 'sma20')  { visSma20 = !visSma20;   $('btn-ind-sma20').classList.toggle('active-sma20', visSma20); }
    if (ind === 'ema50')  { visEma50 = !visEma50;   $('btn-ind-ema50').classList.toggle('active-ema50', visEma50); }
    if (ind === 'sma200') { visSma200 = !visSma200; $('btn-ind-sma200').classList.toggle('active-sma200', visSma200); }
    if (ind === 'vwap')   { visVwap = !visVwap;     $('btn-ind-vwap').classList.toggle('active-vwap', visVwap); }
    if (ind === 'rsi')    { visRsi = !visRsi;       $('btn-ind-rsi').classList.toggle('active-rsi', visRsi); $('rsi-container').style.display = visRsi ? 'block' : 'none'; }
    if (ind === 'sltp')   { visSLTP = !visSLTP;     $('btn-ind-sltp').classList.toggle('active-sltp', visSLTP); renderPriceLines(activeSymbol); }
    applyIndicatorVisibility();
  }

  function applyIndicatorVisibility() {
    if (ema12Series)  ema12Series.applyOptions({ visible: visEma12 });
    if (sma20Series)  sma20Series.applyOptions({ visible: visSma20 });
    if (ema50Series)  ema50Series.applyOptions({ visible: visEma50 });
    if (sma200Series) sma200Series.applyOptions({ visible: visSma200 });
    if (vwapSeries)   vwapSeries.applyOptions({ visible: visVwap });
  }

  function applyZoomRange(rangeType) {
    if (!tvChart || currentCandleData.length === 0) return;
    const totalBars = currentCandleData.length;
    if (rangeType === 'all') {
      tvChart.timeScale().fitContent();
      rsiChart.timeScale().fitContent();
      return;
    }
    let bCount = 90;
    if (rangeType === '1w') bCount = 7;
    else if (rangeType === '1m') bCount = 22;
    else if (rangeType === '3m') bCount = 66;
    else if (rangeType === '6m') bCount = 130;
    else if (rangeType === '1y') bCount = 250;
    else if (rangeType === 'default') bCount = 90;

    const fromIdx = Math.max(0, totalBars - bCount);
    tvChart.timeScale().setVisibleLogicalRange({ from: fromIdx, to: totalBars + 2 });
    rsiChart.timeScale().setVisibleLogicalRange({ from: fromIdx, to: totalBars + 2 });
  }

  function setTimeframe(tf) {
    activeInterval = tf;
    $('tf-1d').classList.toggle('active', tf==='1d');
    $('tf-1h').classList.toggle('active', tf==='1h');
    $('tf-15m').classList.toggle('active', tf==='15m');
    loadChart(activeSymbol);
  }

  let isCollapsed = false;

  function toggleCollapseChart() {
    isCollapsed = !isCollapsed;
    $('chart-body').style.display = isCollapsed ? 'none' : 'block';
    const btn = $('btn-collapse');
    btn.innerHTML = isCollapsed ? '＋ Expand' : '─';
    btn.title = isCollapsed ? 'Expand Chart' : 'Minimize/Collapse Chart';
    if (!isCollapsed && tvChart) {
      setTimeout(() => {
        const w = $('chart-container').clientWidth;
        tvChart.applyOptions({ width: w });
        if (rsiChart) rsiChart.applyOptions({ width: w });
      }, 50);
    }
  }

  function toggleFullscreen() {
    if (isCollapsed) toggleCollapseChart();
    isFullscreen = !isFullscreen;
    $('chart-card').classList.toggle('fullscreen-card', isFullscreen);
    const btn = $('btn-fs');
    if (isFullscreen) {
      btn.innerHTML = '🗗 Minimize (ESC)';
      btn.classList.remove('action-btn-ghost');
      btn.style.background = 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)';
      btn.style.color = '#fff';
    } else {
      btn.innerHTML = '⛶ Maximize';
      btn.classList.add('action-btn-ghost');
      btn.style.background = '';
      btn.style.color = '';
    }
    setTimeout(() => {
      const w = $('chart-container').clientWidth;
      tvChart.applyOptions({ width: w, height: isFullscreen ? (window.innerHeight - 200) : 380 });
      if (rsiChart) rsiChart.applyOptions({ width: w });
    }, 100);
  }

  // ESC Key to exit Fullscreen
  window.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      if (isFullscreen) toggleFullscreen();
    }
  });

  // Shift + Drag Box Zoom
  function initBoxZoom() {
    const cEl = $('chart-container');
    const overlay = $('box-zoom-overlay');
    let startX = null;

    cEl.addEventListener('mousedown', e => {
      if (!e.shiftKey) return;
      startX = e.clientX - cEl.getBoundingClientRect().left;
      overlay.style.left = startX + 'px';
      overlay.style.top = '0px';
      overlay.style.width = '0px';
      overlay.style.height = cEl.clientHeight + 'px';
      overlay.style.display = 'block';
      tvChart.applyOptions({ handleScroll: false, handleScale: false });
    });

    cEl.addEventListener('mousemove', e => {
      if (startX === null) return;
      const currentX = e.clientX - cEl.getBoundingClientRect().left;
      overlay.style.left = Math.min(startX, currentX) + 'px';
      overlay.style.width = Math.abs(currentX - startX) + 'px';
    });

    window.addEventListener('mouseup', e => {
      if (startX === null) return;
      const cRect = cEl.getBoundingClientRect();
      const endX = Math.max(0, Math.min(cRect.width, e.clientX - cRect.left));
      const x1 = Math.min(startX, endX);
      const x2 = Math.max(startX, endX);
      startX = null;
      overlay.style.display = 'none';

      if (x2 - x1 > 15 && tvChart) {
        const timeScale = tvChart.timeScale();
        const t1 = timeScale.coordinateToTime(x1);
        const t2 = timeScale.coordinateToTime(x2);
        if (t1 && t2) {
          timeScale.setVisibleRange({ from: t1, to: t2 });
          rsiChart.timeScale().setVisibleRange({ from: t1, to: t2 });
        }
      }
      tvChart.applyOptions({ handleScroll: true, handleScale: true });
    });
  }

  // Fetch Live Portfolio Data from SQLite API
  async function refreshPortfolio() {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) return;
      const data = await res.json();
      currentPortfolioData = data;

      // Update Financial HUD
      $('nav-val').textContent = `₹${fmt(data.total_nav)}`;
      const navGainPct = ((data.total_nav - 100000) / 100000 * 100).toFixed(2);
      $('nav-gain').textContent = `${navGainPct >= 0 ? '+' : ''}${navGainPct}% All-Time Gain`;
      $('nav-dd').textContent = `${((data.peak_nav - data.total_nav)/data.peak_nav*100).toFixed(2)}%`;
      $('cash-val').textContent = `₹${fmt(data.cash)}`;

      let totalUnreal = 0;
      data.positions.forEach(p => totalUnreal += p.unrealized_pnl);
      $('unreal-val').textContent = `${totalUnreal >= 0 ? '+' : '-'}₹${fmt(Math.abs(totalUnreal))}`;
      $('unreal-val').className = `hud-value ${totalUnreal >= 0 ? 'pos' : 'neg'}`;
      $('unreal-pct').textContent = `${totalUnreal >= 0 ? '+' : ''}${((totalUnreal / (data.invested || 1))*100).toFixed(2)}%`;

      // Update Regime Pill
      const reg = data.active_regime;
      $('regime-badge').textContent = `🛡️ ${reg}`;
      $('shield-val').textContent = reg.replace('_', ' ');

      renderPositionInspector(data.positions);
      renderPriceLines(activeSymbol);
      renderBottomDrawer();
    } catch(e) {}
  }

  function renderPositionInspector(positions) {
    const c = $('position-inspector-content');
    if (!positions || positions.length === 0) {
      c.innerHTML = `
        <div style="text-align:center; padding: 40px 20px; color: var(--text-muted);">
          <div style="font-size:24px; margin-bottom:8px;">🛡️</div>
          <div>No active open positions under CNC Delivery.</div>
          <div style="font-size:11px; margin-top:4px;">100% Capital preserved in Liquid Cash Shield.</div>
        </div>
      `;
      return;
    }

    const p = positions[0]; // Active holding
    const pctToTarget = Math.max(0, Math.min(100, ((p.current - p.stop_loss) / (p.take_profit - p.stop_loss) * 100))).toFixed(1);

    c.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
        <div>
          <div style="font-size:18px; font-weight:800; color:#fff;" class="mono">${p.symbol}</div>
          <div style="font-size:11px; color:var(--text-secondary);">${p.strategy}</div>
        </div>
        <div style="text-align:right;">
          <div class="mono ${p.unrealized_pnl >= 0 ? 'pos' : 'neg'}" style="font-size:16px; font-weight:700;">
            ${p.unrealized_pnl >= 0 ? '+' : ''}₹${fmt(p.unrealized_pnl)}
          </div>
          <div style="font-size:11px;" class="${p.unrealized_pnl >= 0 ? 'pos' : 'neg'}">
            (${p.unrealized_pnl >= 0 ? '+' : ''}${p.unrealized_pnl_pct}%)
          </div>
        </div>
      </div>

      <div class="risk-gauge-box">
        <div style="display:flex; justify-content:space-between; font-size:11px; font-weight:700;">
          <span style="color:#f43f5e;">STOP LOSS: ₹${p.stop_loss}</span>
          <span style="color:#60a5fa;">CURRENT: ₹${p.current}</span>
          <span style="color:#10b981;">TARGET: ₹${p.take_profit}</span>
        </div>
        <div class="risk-meter-track">
          <div class="risk-meter-fill" style="width:${pctToTarget}%;"></div>
          <div class="risk-meter-thumb" style="left:${pctToTarget}%;"></div>
        </div>
        <div class="risk-meter-labels">
          <span>1.25x ATR Risk Floor</span>
          <span>${pctToTarget}% to Target</span>
          <span>3.50x ATR Target Lock</span>
        </div>
      </div>

      <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; font-size:11px; margin-bottom:12px;">
        <div class="pill" style="justify-content:space-between;">
          <span style="color:var(--text-muted);">Shares Held</span>
          <span class="mono" style="font-weight:700;">${p.qty} shs</span>
        </div>
        <div class="pill" style="justify-content:space-between;">
          <span style="color:var(--text-muted);">Entry Price</span>
          <span class="mono" style="font-weight:700;">₹${p.entry}</span>
        </div>
        <div class="pill" style="justify-content:space-between;">
          <span style="color:var(--text-muted);">Invested Capital</span>
          <span class="mono" style="font-weight:700;">₹${fmt(p.entry * p.qty)}</span>
        </div>
        <div class="pill" style="justify-content:space-between;">
          <span style="color:var(--text-muted);">Trailing Stop</span>
          <span style="color:#10b981; font-weight:700;">${p.be_locked ? '🛡️ Break-Even Active' : '📍 Active ATR'}</span>
        </div>
      </div>
    `;
  }

  function switchBottomTab(tab) {
    activeBottomTab = tab;
    $('tab-tr').classList.toggle('active', tab === 'trades');
    $('tab-tg').classList.toggle('active', tab === 'telegram');
    renderBottomDrawer();
  }

  function renderBottomDrawer() {
    const c = $('bottom-content');
    if (!currentPortfolioData) return;

    if (activeBottomTab === 'trades') {
      const trades = currentPortfolioData.trades || [];
      if (trades.length === 0) {
        c.innerHTML = '<div style="text-align:center; padding:30px; color:var(--text-muted);">No completed trade exits yet. Open positions are currently compounding.</div>';
        return;
      }
      let html = `
        <table>
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Strategy</th>
              <th>Exit Date</th>
              <th>Entry Price</th>
              <th>Exit Price</th>
              <th>Qty</th>
              <th>Net P&L (₹)</th>
              <th>Taxes Paid</th>
              <th>Exit Reason</th>
            </tr>
          </thead>
          <tbody>
      `;
      trades.forEach(t => {
        html += `
          <tr>
            <td class="mono" style="font-weight:700;">${t.symbol}</td>
            <td style="color:var(--text-secondary);">${t.strategy}</td>
            <td class="mono">${t.exit_date}</td>
            <td class="mono">₹${t.entry_price}</td>
            <td class="mono">₹${t.exit_price}</td>
            <td class="mono">${t.qty}</td>
            <td class="mono ${t.net_pnl >= 0 ? 'pos' : 'neg'}">${t.net_pnl >= 0 ? '+' : ''}₹${fmt(t.net_pnl)} (${t.net_pnl_pct}%)</td>
            <td class="mono" style="color:var(--text-muted);">₹${t.taxes}</td>
            <td><span class="badge-env">${t.reason}</span></td>
          </tr>
        `;
      });
      html += '</tbody></table>';
      c.innerHTML = html;
    } else {
      // Telegram & System Health View
      c.innerHTML = `
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; padding:10px 0;">
          <div style="background:rgba(255,255,255,0.02); border:1px solid var(--card-border); border-radius:8px; padding:16px;">
            <div style="font-weight:700; color:#fff; margin-bottom:8px;">📱 Connected Telegram Recipients</div>
            <div style="display:flex; flex-direction:column; gap:8px;">
              <div class="pill" style="justify-content:space-between; background:rgba(16,185,129,0.08); border-color:rgba(16,185,129,0.3);">
                <span>👤 Infinity (Primary Admin)</span>
                <span class="mono" style="color:#34d399;">ID: 1292507208 • Active 🟢</span>
              </div>
              <div class="pill" style="justify-content:space-between; background:rgba(16,185,129,0.08); border-color:rgba(16,185,129,0.3);">
                <span>👤 Nikhil Keshav (Family Trader)</span>
                <span class="mono" style="color:#34d399;">ID: 8931985247 • Active 🟢</span>
              </div>
            </div>
          </div>
          <div style="background:rgba(255,255,255,0.02); border:1px solid var(--card-border); border-radius:8px; padding:16px;">
            <div style="font-weight:700; color:#fff; margin-bottom:8px;">⏰ Automated Crontab Scheduler</div>
            <div style="color:var(--text-secondary); font-size:12px; margin-bottom:10px;">
              Runs on <strong>@reboot</strong> and every 15 minutes between 09:00 - 23:59 on weekdays with 0.01s Fast-Skip.
            </div>
            <div class="pill" style="justify-content:space-between;">
              <span>Execution Schedule</span>
              <span class="mono" style="color:#60a5fa;">*/15 9-23 * * 1-5</span>
            </div>
          </div>
        </div>
      `;
    }
  }

  // 1-Click Action Triggers
  async function triggerMarketScan() {
    showToast("⏳ Executing Live Market Scan from NSE...");
    try {
      const res = await fetch('/api/trigger_scan', { method: 'POST' });
      const data = await res.json();
      showToast(`✅ ${data.message || 'Market Scan Completed!'}`);
      refreshPortfolio();
    } catch(e) {
      showToast("❌ Scan execution error");
    }
  }

  async function triggerTestNotify() {
    showToast("📨 Sending test Telegram alert to both phones...");
    try {
      const res = await fetch('/api/test_notify', { method: 'POST' });
      const data = await res.json();
      showToast(`🎉 ${data.message || 'Test alert dispatched!'}`);
    } catch(e) {
      showToast("❌ Notification failed");
    }
  }

  // Clock update
  function updateClock() {
    const now = new Date();
    const ist = new Date(now.toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
    const h = ist.getHours();
    const m = ist.getMinutes();
    const s = ist.getSeconds();
    const day = ist.getDay();

    const isWeekday = (day >= 1 && day <= 5);
    const isOpen = isWeekday && ((h === 9 && m >= 15) || (h > 9 && h < 15) || (h === 15 && m <= 30));

    const p = $('session-pill');
    if (isOpen) {
      p.className = "pill pill-regime bull";
      p.innerHTML = `🟢 NSE OPEN | ${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')} IST`;
    } else {
      p.className = "pill";
      p.innerHTML = `🔴 NSE CLOSED | ${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')} IST`;
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    initChart();
    refreshPortfolio();
    updateClock();
    setInterval(updateClock, 1000);
    setInterval(refreshPortfolio, 10000); // 10s auto-refresh
  });
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────
#  HTTP Request Handler & API Endpoints
# ─────────────────────────────────────────────────────────────────

class ProCockpitHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))

        elif self.path == "/api/status":
            data = get_live_portfolio_data()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))

        elif self.path.startswith("/api/candles"):
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            sym = qs.get("symbol", ["GOLDBEES.NS"])[0]
            interval = qs.get("interval", ["1d"])[0]
            payload = fetch_candles_for_chart(sym, interval)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/trigger_scan":
            def _scan():
                subprocess.run([str(ROOT_DIR / ".venv" / "bin" / "python"), str(ROOT_DIR / "run_live_paper_orchestrator.py"), "--force"])

            threading.Thread(target=_scan, daemon=True).start()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "message": "Live market scan started. Updating quotes..."}).encode("utf-8"))

        elif self.path == "/api/test_notify":
            notifier = Notifier()
            success = notifier.notify_trade_entry("GOLDBEES.NS", "Sovereign Gold Defense Shield", 133.02, 723, 124.75, 164.15, 5.0, "BEAR_DEFENSE")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "message": "Test alert sent to Infinity & Nikhil Keshav!" if success else "Notification error"}).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass


def main():
    port = 8080
    server = HTTPServer(("0.0.0.0", port), ProCockpitHandler)
    print(f"🚀 State-of-the-Art Pro-Trader Cockpit v4.0 is LIVE at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        server.server_close()


if __name__ == "__main__":
    main()
