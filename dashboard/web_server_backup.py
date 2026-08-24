# ─────────────────────────────────────────────────────────────────
#  dashboard/web_server.py
#
#  Lightweight HTTP dashboard — zero extra dependencies.
#  Uses Python's built-in http.server + json modules only.
#
#  Serves:
#    GET /          → live HTML dashboard (auto-refreshes every 5s)
#    GET /api/status → JSON snapshot of portfolio, prices, signals
#
#  Run in a daemon thread so it never blocks the main trading loop.
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import threading
from datetime             import datetime, timezone
from http.server          import BaseHTTPRequestHandler, HTTPServer
from typing               import Optional

from engine.portfolio     import Portfolio
from engine.ledger        import Ledger
from config.india_settings  import INDIA_UI_DEFAULT_VISIBLE_BARS


# ─────────────────────────────────────────────────────────────────
#  Shared state  (written by main loop, read by HTTP handler)
# ─────────────────────────────────────────────────────────────────

class BotState:
    """Thread-safe snapshot of the bot's current state."""
    def __init__(self):
        self.prices:      dict[str, float]  = {}
        self.prev_prices: dict[str, float]  = {}
        self.signals:     dict              = {}
        self.portfolio:   Optional[Portfolio] = None
        self.ledger:      Optional[Ledger]  = None
        self.start_time:  datetime          = datetime.now(timezone.utc)
        self.last_update: str               = "—"
        self._lock = threading.Lock()

    def update(self, prices, prev_prices, signals, portfolio, ledger):
        with self._lock:
            self.prices      = dict(prices)
            self.prev_prices = dict(prev_prices)
            self.signals     = signals
            self.portfolio   = portfolio
            self.ledger      = ledger
            self.last_update = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    def _india_status(self) -> dict:
        import os, sqlite3, json
        res = {"cash": 0, "total_value": 0, "total_pnl": 0, "positions": [], "trades": [], "prices": {}}
        db_path = "india_paper.sqlite"
        if not os.path.exists(db_path):
            return res
        try:
            with sqlite3.connect(db_path, uri=True) as conn:
                conn.row_factory = sqlite3.Row
                cash_row = conn.execute("SELECT cash FROM state WHERE id=1").fetchone()
                cash = cash_row["cash"] if cash_row else 0.0
                res["cash"] = round(cash, 2)
                res["total_value"] = cash

                # Extract live/latest prices from local daily charts table
                ind_prices = {}
                try:
                    c_rows = conn.execute("SELECT symbol, data_json FROM charts").fetchall()
                    for c_sym, c_json in c_rows:
                        c_data = json.loads(c_json)
                        if len(c_data) >= 2:
                            ltp = c_data[-1]["close"]
                            prev = c_data[-2]["close"]
                            chg = ((ltp - prev) / prev * 100.0) if prev else 0.0
                            ind_prices[c_sym] = {"price": round(ltp, 2), "change_pct": round(chg, 3)}
                except Exception:
                    pass
                res["prices"] = ind_prices

                pos_rows = conn.execute("SELECT * FROM positions").fetchall()
                for r in pos_rows:
                    sym = r["symbol"]
                    entry = r["entry_price"]
                    qty = r["qty"]
                    ltp = ind_prices.get(sym, {}).get("price", entry)
                    pnl = (ltp - entry) * qty
                    pnl_pct = ((ltp - entry) / entry * 100.0) if entry else 0.0
                    res["positions"].append({
                        "symbol": sym,
                        "entry": round(entry, 2),
                        "current": round(ltp, 2),
                        "qty": round(qty, 2),
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "sl": round(r["sl"], 2),
                        "tp": round(r["tp"], 2)
                    })
                    res["total_value"] += (entry * qty) + pnl

                tr_rows = conn.execute("SELECT * FROM history ORDER BY id DESC LIMIT 10").fetchall()
                total_pnl = conn.execute("SELECT SUM(pnl) as tpnl FROM history").fetchone()["tpnl"] or 0
                res["total_pnl"] = round(total_pnl, 2)

                for r in tr_rows:
                    res["trades"].append({
                        "time": r["exit_time"][:16].replace("T", " "),
                        "symbol": r["symbol"],
                        "type": r["exit_type"],
                        "pnl": round(r["pnl"], 2),
                        "is_win": r["pnl"] > 0
                    })
        except Exception:
            pass
        return res

    def snapshot(self) -> dict:
        with self._lock:
            p = self.portfolio
            prices = self.prices

            # Portfolio
            total_val = p.total_value(prices) if p else 0
            total_pnl = p.total_pnl(prices)   if p else 0
            pnl_pct   = p.total_pnl_pct(prices) if p else 0

            # Positions
            positions = []
            if p:
                for pos in p.positions.values():
                    price = prices.get(pos.symbol, pos.entry_price)
                    pnl   = pos.unrealized_pnl(price)
                    positions.append({
                        "id":         pos.id,
                        "symbol":     pos.symbol,
                        "direction":  pos.direction,
                        "entry":      pos.entry_price,
                        "current":    price,
                        "stop_loss":  pos.stop_loss,
                        "tp1":        pos.take_profit1,
                        "tp2":        pos.take_profit2,
                        "pnl":        round(pnl, 2),
                        "pnl_pct":    round(pos.unrealized_pnl_pct(price), 2),
                        "prob":       pos.probability,
                        "tp1_hit":    False,
                        "breakeven":  False,
                    })

            # Trades
            trades = []
            if self.ledger:
                for r in reversed(self.ledger.recent(10)):
                    trades.append({
                        "time":       r.exit_time.strftime("%H:%M"),
                        "symbol":     r.symbol,
                        "direction":  r.direction,
                        "type":       r.close_type,
                        "entry":      r.entry_price,
                        "exit":       r.exit_price,
                        "pnl":        round(r.pnl, 4),
                        "is_win":     r.is_win,
                    })

            # Signals
            signals_data = {}
            for sym, sig in self.signals.items():
                if sig:
                    signals_data[sym] = {
                        "probability": sig.probability,
                        "direction":   sig.direction,
                        "confidence":  sig.confidence,
                    }
                else:
                    signals_data[sym] = None

            # Prices with % change
            prices_data = {}
            for sym, price in prices.items():
                prev  = self.prev_prices.get(sym, price)
                chg   = ((price - prev) / prev * 100) if prev else 0
                prices_data[sym] = {"price": price, "change_pct": round(chg, 3)}

            india_data = self._india_status()
            # Merge India prices into top prices box
            if "prices" in india_data:
                for ind_sym, pinfo in india_data["prices"].items():
                    prices_data[ind_sym] = pinfo

            elapsed = int((datetime.now(timezone.utc) - self.start_time).total_seconds())
            h, rem  = divmod(elapsed, 3600)
            m, s    = divmod(rem, 60)

            return {
                "india": india_data,
                "portfolio": {
                    "total_value":   round(total_val, 2),
                    "cash":          round(p.cash, 2) if p else 0,
                    "pnl":           round(total_pnl, 2),
                    "pnl_pct":       round(pnl_pct, 2),
                    "win_rate":      round(p.win_rate(), 1) if p else 0,
                    "total_trades":  p.total_trades if p else 0,
                    "wins":          p.winning_trades if p else 0,
                    "open_trades":   len(positions),
                },
                "prices":      prices_data,
                "signals":     signals_data,
                "positions":   positions,
                "trades":      trades,
                "runtime":     f"{h:02d}:{m:02d}:{s:02d}",
                "last_update": self.last_update,
            }


# ─────────────────────────────────────────────────────────────────
#  HTML page  (inline, no external files needed)
# ─────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AutoTrader — Paper Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://unpkg.com/lightweight-charts@3.8.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root {
    --bg:      #0a0e1a;
    --surface: #111827;
    --border:  #1f2937;
    --accent:  #3b82f6;
    --green:   #10b981;
    --red:     #ef4444;
    --yellow:  #f59e0b;
    --text:    #e5e7eb;
    --muted:   #6b7280;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; font-size: 14px; }
  .mono { font-family: 'JetBrains Mono', monospace; }

  /* Header */
  header { background: linear-gradient(135deg, #1e3a5f 0%, #111827 100%);
    border-bottom: 1px solid var(--border); padding: 14px 24px;
    display: flex; align-items: center; justify-content: space-between; }
  header .brand { display: flex; align-items: center; gap: 10px; }
  header .brand span { font-size: 20px; font-weight: 700; color: #fff; }
  header .badge { background: #f59e0b22; color: var(--yellow); border: 1px solid var(--yellow);
    padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; letter-spacing: 1px; }
  header .meta { color: var(--muted); font-size: 12px; text-align: right; line-height: 1.6; }

  /* Layout */
  .container { max-width: 1600px; margin: 0 auto; padding: 20px; }
  .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 16px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }

  /* Cards */
  .card { background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 18px; }
  .card-title { color: var(--muted); font-size: 11px; font-weight: 600;
    letter-spacing: 1px; text-transform: uppercase; margin-bottom: 14px;
    border-bottom: 1px solid var(--border); padding-bottom: 8px; }

  /* Stats grid inside card */
  .stat-row { display: flex; justify-content: space-between; align-items: center; padding: 5px 0; }
  .stat-label { color: var(--muted); font-size: 12px; }
  .stat-value { font-family: 'JetBrains Mono', monospace; font-weight: 600; }

  /* Colour helpers */
  .pos { color: var(--green); }
  .neg { color: var(--red); }
  .neu { color: var(--muted); }
  .acc { color: var(--accent); }
  .yel { color: var(--yellow); }

  /* Indicator toggle buttons */
  .ind-btn { background: #1f2937; border: 1px solid #374151; color: #9ca3af; padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; cursor: pointer; transition: all 0.15s ease; font-family: 'JetBrains Mono', monospace; }
  .ind-btn:hover { border-color: #4b5563; color: #f3f4f6; }
  .ind-btn.active-ema12 { background: #3b82f622; border-color: #3b82f6; color: #60a5fa; }
  .ind-btn.active-ema50 { background: #f59e0b22; border-color: #f59e0b; color: #fbbf24; }
  .ind-btn.active-vwap  { background: #e5e7eb22; border-color: #e5e7eb; color: #f9fafb; }
  .ind-btn.active-box   { background: #10b98122; border-color: #10b981; color: #34d399; }
  #box-zoom-overlay { position: absolute; background: rgba(16, 185, 129, 0.15); border: 1px dashed #10b981; pointer-events: none; display: none; z-index: 50; }

  /* Prices */
  .price-row { display: flex; justify-content: space-between; padding: 8px 0;
    border-bottom: 1px solid var(--border); align-items: center; }
  .price-row:last-child { border-bottom: none; }
  .price-sym { font-weight: 700; font-size: 13px; }
  .price-val { font-family: 'JetBrains Mono', monospace; font-size: 15px; font-weight: 600; }
  .price-chg { font-size: 12px; font-family: 'JetBrains Mono', monospace; }

  /* Signal badge */
  .sig-row { display: flex; justify-content: space-between; padding: 8px 0;
    border-bottom: 1px solid var(--border); align-items: center; }
  .sig-row:last-child { border-bottom: none; }
  .sig-dir { padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }
  .dir-LONG     { background: #10b98122; color: var(--green); border: 1px solid var(--green); }
  .dir-SHORT    { background: #ef444422; color: var(--red);   border: 1px solid var(--red); }
  .dir-NO_TRADE { background: #6b728022; color: var(--muted); border: 1px solid var(--muted); }

  /* Tables */
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { color: var(--muted); font-weight: 600; font-size: 11px; letter-spacing: 0.5px;
       text-transform: uppercase; padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--border); }
  td { padding: 9px 10px; border-bottom: 1px solid var(--border); vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #ffffff05; }
  .mono-td { font-family: 'JetBrains Mono', monospace; }

  /* Status bar */
  .status-bar { text-align: center; color: var(--muted); font-size: 12px; margin-top: 8px;
    padding: 10px; background: var(--surface); border-radius: 8px; border: 1px solid var(--border); }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: var(--green); margin-right: 6px; animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span>🤖 AutoTrader</span>
    <div class="badge">PAPER TRADING</div>
  </div>
  <div class="meta" id="header-meta">Loading...</div>
</header>

<div class="container">

  <!-- Row 1: Portfolio / Prices / Signals -->
  <div class="grid-3">

    <div class="card">
      <div class="card-title">Portfolio</div>
      <div class="metric-big acc" id="total-value">$—</div>
      <div class="metric-sub" id="pnl-line">—</div>
      <br>
      <div class="stat-row"><span class="stat-label">Cash</span>         <span class="stat-value mono" id="cash">—</span></div>
      <div class="stat-row"><span class="stat-label">Win Rate</span>     <span class="stat-value yel" id="win-rate">—</span></div>
      <div class="stat-row"><span class="stat-label">Total Trades</span> <span class="stat-value" id="total-trades">—</span></div>
      <div class="stat-row"><span class="stat-label">Open Positions</span><span class="stat-value acc" id="open-trades">—</span></div>
    </div>

    <div class="card">
      <div class="card-title">Live Prices</div>
      <div id="prices-container">Loading...</div>
    </div>

    <div class="card">
      <div class="card-title">Signals</div>
      <div id="signals-container">Loading...</div>
    </div>
  </div>

  <!-- Row 1.5: India Equities -->
  <div class="grid-2">
    <div class="card">
      <div class="card-title">🇮🇳 INDIA EQUITIES (CNC DELIVERY)</div>
      <div class="grid-2" style="margin-bottom: 0;">
        <div>
            <div class="stat-row"><span class="stat-label">Total Value</span> <span class="stat-value mono acc" id="ind-value">₹—</span></div>
            <div class="stat-row"><span class="stat-label">Available Cash</span> <span class="stat-value mono" id="ind-cash">₹—</span></div>
            <div class="stat-row"><span class="stat-label">Realized P&L</span> <span class="stat-value mono" id="ind-pnl">₹—</span></div>
        </div>
        <div>
            <table style="margin-top: 5px;">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Entry</th>
                  <th>LTP</th>
                  <th>P&L</th>
                  <th>SL / TP</th>
                </tr>
              </thead>
              <tbody id="ind-pos-body">
                <tr><td colspan="5" class="neu" style="text-align:center">No open positions</td></tr>
              </tbody>
            </table>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">LATEST INDIA TRADES</div>
        <table>
          <tbody id="ind-trades-body">
            <tr><td colspan="4" class="neu" style="text-align:center">No completed trades yet</td></tr>
          </tbody>
        </table>
    </div>
  </div>

  <!-- Row 1.7: India TV Chart -->
  <div class="card" style="margin-bottom:16px;">
    <div class="card-title" style="display:flex; justify-content:space-between; align-items:center;">
      <span>INDIA MARKET CHART</span>
      <div style="display:flex; gap:8px;">
        <select id="chart-sym-select" style="background:#1f2937; color:#e5e7eb; border:none; border-radius:4px; padding:3px 8px; font-family:'JetBrains Mono',monospace;">
           <option value="^NSEI">^NSEI (Nifty 50)</option>
           <option value="INFY.NS">INFY.NS</option>
           <option value="TCS.NS">TCS.NS</option>
           <option value="LT.NS">LT.NS</option>
           <option value="RELIANCE.NS">RELIANCE.NS</option>
           <option value="BHARTIARTL.NS">BHARTIARTL.NS</option>
           <option value="SBIN.NS">SBIN.NS</option>
           <option value="ICICIBANK.NS">ICICIBANK.NS</option>
        </select>
        <select id="chart-tf-select" style="background:#3b82f6; color:#ffffff; font-weight:700; border:none; border-radius:4px; padding:3px 8px; font-family:'JetBrains Mono',monospace;">
           <option value="1d">Daily (1D)</option>
           <option value="1h">Hourly (1H)</option>
           <option value="15m">15 Min (15M)</option>
        </select>
        <button id="btn-ind-ema12" class="ind-btn active-ema12" onclick="toggleInd('ema12')">EMA12</button>
        <button id="btn-ind-ema50" class="ind-btn active-ema50" onclick="toggleInd('ema50')">EMA50</button>
        <button id="btn-ind-vwap" class="ind-btn active-vwap" onclick="toggleInd('vwap')">VWAP</button>
        <span style="font-size:11px; color:#10b981; font-weight:600; background:#10b98118; padding:3px 8px; border-radius:4px; border:1px solid #10b98144;" title="Hold Shift and drag mouse on chart to box-zoom any range instantly">💡 Hold SHIFT + Drag to Zoom</span>
        <span style="color:#4b5563">&nbsp;|&nbsp;</span>
        <select id="chart-zoom-select" onchange="applyZoomRange(this.value)" style="background:#1f2937; color:#10b981; font-weight:700; border:1px solid #374151; border-radius:4px; padding:3px 8px; font-family:'JetBrains Mono',monospace;">
           <option value="default">Zoom: Default</option>
           <option value="1w">Zoom: 1 Week</option>
           <option value="1m">Zoom: 1 Month</option>
           <option value="3m">Zoom: 3 Months</option>
           <option value="6m">Zoom: 6 Months</option>
           <option value="1y">Zoom: 1 Year</option>
           <option value="all">Zoom: ALL</option>
        </select>
      </div>
    </div>
    <div style="position: relative; width: 100%;">
      <div id="tv-chart" style="width: 100%; height: 350px;"></div>
      <div id="box-zoom-overlay"></div>
    </div>
  </div>

  <!-- Row 2: Open Positions -->
  <div class="card" style="margin-bottom:16px">
    <div class="card-title">Open Positions</div>
    <table>
      <thead>
        <tr>
          <th>ID</th><th>Symbol</th><th>Dir</th><th>Entry</th><th>Current</th>
          <th>Stop Loss</th><th>TP1</th><th>TP2</th><th>Unrealized P&L</th><th>Prob</th><th>Status</th>
        </tr>
      </thead>
      <tbody id="positions-body"><tr><td colspan="11" class="neu">No open positions</td></tr></tbody>
    </table>
  </div>

  <!-- Row 3: Recent Trades -->
  <div class="card" style="margin-bottom:16px">
    <div class="card-title">Recent Trades</div>
    <table>
      <thead>
        <tr><th>Time</th><th>Symbol</th><th>Dir</th><th>Type</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Result</th></tr>
      </thead>
      <tbody id="trades-body"><tr><td colspan="8" class="neu">No completed trades yet</td></tr></tbody>
    </table>
  </div>

  <div class="status-bar">
    <span class="dot"></span>
    <span id="status-text">Connecting to bot...</span>
  </div>
</div>

<script>
  const $ = id => document.getElementById(id);
  const fmt = n => n.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2});
  const fmtS = n => (n >= 0 ? '+' : '') + fmt(n);
  const color = (n, pos='pos', neg='neg') => n >= 0 ? pos : neg;

  async function refresh() {
    // 1. Fetch India Chart Data independently
    try {
      const chartRes = await fetch('/api/india_charts');
      if (chartRes.ok) {
         const fetched1d = await chartRes.json();
         // Update only 1d keys so selected 1h/15m chart data doesn't get wiped/overwritten
         for (const k in fetched1d) { chartDataMap[k] = fetched1d[k]; }
         if(tvChart) renderSelectedChart(false);
      }
    } catch(ce) { console.warn("India chart data fetch failed."); }

    // 2. Fetch Main Status
    try {
      const r = await fetch('/api/status');
      if (!r.ok) return;
      const d = await r.json();
      const p = d.portfolio;

      // Header
      $('header-meta').innerHTML = `Runtime: <strong>${d.runtime}</strong> &nbsp;|&nbsp; Last update: <strong>${d.last_update}</strong>`;

      // Portfolio
      $('total-value').textContent = '$' + fmt(p.total_value);
      const pnlSign = p.pnl >= 0 ? '+' : '';
      $('pnl-line').innerHTML = `<span class="${color(p.pnl)}">${pnlSign}$${fmt(p.pnl)} (${pnlSign}${p.pnl_pct}%)</span>`;
      $('cash').textContent = '$' + fmt(p.cash);
      $('win-rate').textContent = `${p.win_rate}%  (${p.wins}W / ${p.total_trades - p.wins}L)`;
      $('total-trades').textContent = p.total_trades;
      $('open-trades').textContent = p.open_trades;

      // Prices
      let priceHTML = '';
      for (const [sym, data] of Object.entries(d.prices)) {
        const chg = data.change_pct;
        const arrow = chg >= 0 ? '▲' : '▼';
        const cls = chg >= 0 ? 'pos' : 'neg';
        priceHTML += `<div class="price-row">
          <span class="price-sym">${sym.replace('USDT','/USDT')}</span>
          <span class="price-val acc mono">$${fmt(data.price)}</span>
          <span class="price-chg ${cls}">${arrow} ${(chg >= 0 ? '+' : '')}${chg.toFixed(3)}%</span>
        </div>`;
      }
      $('prices-container').innerHTML = priceHTML || '<span class="neu">No data</span>';

      // Signals
      let sigHTML = '';
      for (const [sym, sig] of Object.entries(d.signals)) {
        const name = sym.replace('USDT','/USDT');
        if (!sig) {
          sigHTML += `<div class="sig-row"><span class="price-sym">${name}</span><span class="neu">Scanning...</span></div>`;
        } else {
          const pct = sig.probability.toFixed(1);
          const pCls = sig.probability >= 70 ? 'pos' : (sig.probability <= 30 ? 'neg' : 'yel');
          sigHTML += `<div class="sig-row">
            <span class="price-sym">${name}</span>
            <span class="${pCls} mono">${pct}%</span>
            <span class="sig-dir dir-${sig.direction}">${sig.direction}</span>
          </div>`;
        }
      }
      $('signals-container').innerHTML = sigHTML || '<span class="neu">No signals</span>';

      // Positions
      if (d.positions.length === 0) {
        $('positions-body').innerHTML = '<tr><td colspan="11" class="neu" style="text-align:center">No open positions</td></tr>';
      } else {
        $('positions-body').innerHTML = d.positions.map(pos => {
          const pCls = color(pos.pnl);
          const dCls = pos.direction === 'LONG' ? 'pos' : 'neg';
          const status = '🔄 OPEN';
          return `<tr>
            <td class="mono neu">${pos.id}</td>
            <td><strong>${pos.symbol.replace('USDT','/USDT')}</strong></td>
            <td class="${dCls} mono-td">${pos.direction}</td>
            <td class="mono-td">$${fmt(pos.entry)}</td>
            <td class="mono-td acc">$${fmt(pos.current)}</td>
            <td class="mono-td neg">$${fmt(pos.stop_loss)}</td>
            <td class="mono-td">$${fmt(pos.tp1)}</td>
            <td class="mono-td">$${fmt(pos.tp2)}</td>
            <td class="mono-td ${pCls}">${fmtS(pos.pnl)} (${fmtS(pos.pnl_pct)}%)</td>
            <td class="mono-td yel">${pos.prob}%</td>
            <td>${status}</td>
          </tr>`;
        }).join('');
      }

      // Trades
      if (d.trades.length === 0) {
        $('trades-body').innerHTML = '<tr><td colspan="8" class="neu" style="text-align:center">No completed trades yet</td></tr>';
      } else {
        $('trades-body').innerHTML = d.trades.map(t => {
          const pCls = t.is_win ? 'pos' : 'neg';
          const dCls = t.direction === 'LONG' ? 'pos' : 'neg';
          const result = t.is_win ? '✅ WIN' : '❌ LOSS';
          return `<tr>
            <td class="neu">${t.time}</td>
            <td><strong>${t.symbol.replace('USDT','/USDT')}</strong></td>
            <td class="${dCls} mono-td">${t.direction}</td>
            <td class="neu">${t.type.replace(/_/g,' ')}</td>
            <td class="mono-td">$${fmt(t.entry)}</td>
            <td class="mono-td">$${fmt(t.exit)}</td>
            <td class="mono-td ${pCls}">${fmtS(t.pnl)}</td>
            <td class="${pCls}">${result}</td>
          </tr>`;
        }).join('');
      }

      // India Portfolio
      if(d.india && d.india.cash > 0) {
        $('ind-value').innerHTML = '₹' + fmt(d.india.total_value);
        $('ind-cash').innerHTML = '₹' + fmt(d.india.cash);
        const indPnlCls = d.india.total_pnl >= 0 ? 'pos' : 'neg';
        $('ind-pnl').innerHTML = `<span class="${indPnlCls}">${fmtS(d.india.total_pnl)}</span>`;

        if (d.india.positions.length === 0) {
          $('ind-pos-body').innerHTML = '<tr><td colspan="5" class="neu" style="text-align:center">No open positions</td></tr>';
        } else {
          $('ind-pos-body').innerHTML = d.india.positions.map(pos => {
            const pCls = pos.pnl >= 0 ? 'pos' : 'neg';
            return `<tr>
              <td><strong>${pos.symbol}</strong></td>
              <td class="mono-td">₹${fmt(pos.entry)}</td>
              <td class="mono-td acc">₹${fmt(pos.current)}</td>
              <td class="mono-td ${pCls}">${fmtS(pos.pnl)} (${pos.pnl_pct >= 0 ? '+' : ''}${pos.pnl_pct}%)</td>
              <td class="mono-td" style="font-size: 11px;"><span class="neg">₹${pos.sl}</span><br><span class="pos">₹${pos.tp}</span></td>
            </tr>`;
          }).join('');
        }

        if (d.india.trades.length === 0) {
          $('ind-trades-body').innerHTML = '<tr><td colspan="4" class="neu" style="text-align:center">No completed trades yet</td></tr>';
        } else {
          $('ind-trades-body').innerHTML = d.india.trades.map(t => {
            const pCls = t.is_win ? 'pos' : 'neg';
            return `<tr>
              <td class="neu" style="font-size: 11px;">${t.time}</td>
              <td><strong>${t.symbol}</strong></td>
              <td class="neu" style="font-size: 10px;">${t.type.replace(/_/g,' ')}</td>
              <td class="mono-td ${pCls}">${fmtS(t.pnl)}</td>
            </tr>`;
          }).join('');
        }
      }

      $('status-text').textContent = `Live · Bot running · Last refresh: ${new Date().toLocaleTimeString()}`;
    } catch(e) {
      $('status-text').textContent = `⚠️ Connection lost — retrying... (${e.message})`;
    }
  }

  refresh();
  setInterval(refresh, 5000);   // auto-refresh every 5 seconds

  // TradingView Lightweight Charts Initialization
  let tvChart = null;
  let candleSeries = null;
  let vwapSeries = null;
  let ema12Series = null;
  let ema50Series = null;
  let chartDataMap = {};

  async function initChart() {
    try {
      const res = await fetch('/api/india_charts?interval=1d');
      chartDataMap = await res.json();

      const chartProps = {
        layout: { background: { type: 'solid', color: '#111827' }, textColor: '#6b7280' },
        grid: { vertLines: { color: '#1f2937' }, horzLines: { color: '#1f2937' } },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { borderColor: '#1f2937' },
        timeScale: { borderColor: '#1f2937', timeVisible: false, secondsVisible: false },
      };

      tvChart = LightweightCharts.createChart($('tv-chart'), chartProps);

      candleSeries = tvChart.addCandlestickSeries({
        upColor: '#10b981', downColor: '#ef4444', borderVisible: false,
        wickUpColor: '#10b981', wickDownColor: '#ef4444'
      });
      ema12Series = tvChart.addLineSeries({ color: '#3b82f6', lineWidth: 1, title: 'EMA12' });
      ema50Series = tvChart.addLineSeries({ color: '#f59e0b', lineWidth: 1, title: 'EMA50' });
      vwapSeries = tvChart.addLineSeries({ color: '#e5e7eb', lineWidth: 2, title: 'VWAP', lineStyle: 2 });

      renderSelectedChart(true);
    } catch(e) {
      console.error('Failed to load chart data:', e);
    }
  }

  async function renderSelectedChart(forceFit = false) {
    if (!tvChart || !candleSeries) return;

    const sym = $('chart-sym-select').value;
    const tf = $('chart-tf-select').value;
    const cacheKey = tf === '1d' ? sym : (sym + '_' + tf);

    tvChart.applyOptions({
      timeScale: { timeVisible: tf !== '1d', secondsVisible: false }
    });

    if (!chartDataMap[cacheKey]) {
      try {
        const r = await fetch(`/api/india_charts?symbol=${encodeURIComponent(sym)}&interval=${tf}`);
        if (r.ok) {
          const fetched = await r.json();
          chartDataMap[cacheKey] = fetched[sym] || [];
        }
      } catch(e) { console.warn("TF fetch fault", e); }
    }

    const data = chartDataMap[cacheKey];
    if (!data || !data.length) {
      console.warn("No chart data for", sym, tf);
      candleSeries.setData([]);
      ema12Series.setData([]);
      ema50Series.setData([]);
      vwapSeries.setData([]);
      return;
    }

    // Deduplicate and strictly sort ascending by time for TradingView Lightweight Charts
    const uniqueMap = new Map();
    data.forEach(d => { uniqueMap.set(d.time, d); });
    const sortedData = Array.from(uniqueMap.values()).sort((a,b) => a.time - b.time);

    try {
      candleSeries.setData(sortedData.map(d => ({ time: d.time, open: d.open, high: d.high, low: d.low, close: d.close })));
      ema12Series.setData(sortedData.filter(d => d.ema12 !== null && d.ema12 !== undefined).map(d => ({ time: d.time, value: d.ema12 })));
      ema50Series.setData(sortedData.filter(d => d.ema50 !== null && d.ema50 !== undefined).map(d => ({ time: d.time, value: d.ema50 })));
      vwapSeries.setData(sortedData.filter(d => d.vwap !== null && d.vwap !== undefined).map(d => ({ time: d.time, value: d.vwap })));
      if (forceFit) {
        try {
          const cfgRes = await fetch('/api/config_views');
          if (cfgRes.ok) {
             const cfgBars = await cfgRes.json();
             const barsToShow = cfgBars[tf] || 22;
             if (sortedData.length > barsToShow) {
                const startBar = sortedData[sortedData.length - barsToShow];
                const endBar = sortedData[sortedData.length - 1];
                tvChart.timeScale().setVisibleRange({ from: startBar.time, to: endBar.time });
                return;
             }
          }
        } catch(ce) { console.warn("cfg view fault", ce); }
        tvChart.timeScale().fitContent();
      }
    } catch(e) {
      console.error("TV Setter Error:", e);
    }
  }

  let ema12Visible = true;
  let ema50Visible = true;
  let vwapVisible = true;

  function toggleInd(ind) {
    if (!tvChart) return;
    if (ind === 'ema12' && ema12Series) {
      ema12Visible = !ema12Visible;
      ema12Series.applyOptions({ visible: ema12Visible });
      $('btn-ind-ema12').classList.toggle('active-ema12', ema12Visible);
    } else if (ind === 'ema50' && ema50Series) {
      ema50Visible = !ema50Visible;
      ema50Series.applyOptions({ visible: ema50Visible });
      $('btn-ind-ema50').classList.toggle('active-ema50', ema50Visible);
    } else if (ind === 'vwap' && vwapSeries) {
      vwapVisible = !vwapVisible;
      vwapSeries.applyOptions({ visible: vwapVisible });
      $('btn-ind-vwap').classList.toggle('active-vwap', vwapVisible);
    }
  }

  function applyZoomRange(rangeType) {
    if (!tvChart || !candleSeries) return;
    if (rangeType === 'all') {
      tvChart.timeScale().fitContent();
      return;
    }
    if (rangeType === 'default') {
      renderSelectedChart(true);
      return;
    }
    const tf = $('chart-tf-select').value;
    let bCount = 22;
    if (rangeType === '1w') {
      bCount = tf === '15m' ? 175 : (tf === '1h' ? 35 : 5);
    } else if (rangeType === '1m') {
      bCount = tf === '15m' ? 750 : (tf === '1h' ? 150 : 22);
    } else if (rangeType === '3m') {
      bCount = tf === '15m' ? 2250 : (tf === '1h' ? 450 : 66);
    } else if (rangeType === '6m') {
      bCount = tf === '15m' ? 4500 : (tf === '1h' ? 900 : 130);
    } else if (rangeType === '1y') {
      bCount = tf === '15m' ? 9000 : (tf === '1h' ? 1800 : 252);
    }

    const sym = $('chart-sym-select').value;
    const cacheKey = tf === '1d' ? sym : (sym + '_' + tf);
    const data = chartDataMap[cacheKey] || [];
    if (data.length > bCount) {
       const startBar = data[data.length - bCount];
       const endBar = data[data.length - 1];
       tvChart.timeScale().setVisibleRange({ from: startBar.time, to: endBar.time });
    } else {
       tvChart.timeScale().fitContent();
    }
  }

  let startX = null;

  document.addEventListener('DOMContentLoaded', () => {
    const cEl = $('tv-chart');
    const overlay = $('box-zoom-overlay');

    cEl.addEventListener('mousedown', e => {
      // Hold exact Shift key anywhere on chart area to trigger native box-zoom
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
      const left = Math.min(startX, currentX);
      const width = Math.abs(currentX - startX);
      overlay.style.left = left + 'px';
      overlay.style.width = width + 'px';
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
        }
      }
      tvChart.applyOptions({ handleScroll: true, handleScale: true });
    });
  });

  $('chart-sym-select').addEventListener('change', () => renderSelectedChart(true));
  $('chart-tf-select').addEventListener('change', () => renderSelectedChart(true));

  // Resize observer
  new ResizeObserver(entries => {
    if (tvChart && entries.length > 0) {
      tvChart.applyOptions({ width: entries[0].contentRect.width, height: 350 });
    }
  }).observe($('tv-chart'));

  initChart();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────
#  HTTP handler
# ─────────────────────────────────────────────────────────────────

def _make_handler(state: BotState):
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/status":
                data = json.dumps(state.snapshot()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/api/config_views":
                data = json.dumps(INDIA_UI_DEFAULT_VISIBLE_BARS).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            elif self.path.startswith("/api/india_charts"):
                import os, sqlite3, urllib.parse
                parsed_url = urllib.parse.urlparse(self.path)
                qs = urllib.parse.parse_qs(parsed_url.query)
                req_sym = qs.get("symbol", [None])[0]
                req_tf = qs.get("interval", ["1d"])[0]

                res = {}
                if os.path.exists("india_paper.sqlite"):
                    try:
                        with sqlite3.connect("india_paper.sqlite", uri=True) as conn:
                            # If specific interval requested (or if we want all symbols for that TF)
                            if req_tf != "1d" and req_sym:
                                rows = conn.execute("""
                                    SELECT timestamp, open, high, low, close, volume
                                    FROM india_ohlcv
                                    WHERE symbol = ? AND interval = ?
                                    ORDER BY timestamp ASC
                                """, (req_sym, req_tf)).fetchall()
                                candles = []
                                closes = []
                                for r in rows:
                                    closes.append(float(r[4]))
                                    # Convert 1h/15m full timestamps to epoch seconds for TradingView Lightweight Charts
                                    raw_ts = r[0]
                                    if req_tf != "1d" and ("-" in str(raw_ts)) and (":" in str(raw_ts)):
                                        import datetime
                                        try:
                                            dt = datetime.datetime.strptime(str(raw_ts)[:19], "%Y-%m-%d %H:%M:%S")
                                            raw_ts = int(dt.timestamp() + 19800) # +5:30 IST offset to display exact local bar time
                                        except Exception:
                                            pass
                                    candles.append({
                                        "time": raw_ts, "open": float(r[1]), "high": float(r[2]),
                                        "low": float(r[3]), "close": float(r[4]), "ema12": None, "ema50": None, "vwap": None
                                    })
                                # Calculate dynamic EMA12 and EMA50 on the fly
                                if candles:
                                    alpha12 = 2.0 / (12 + 1)
                                    alpha50 = 2.0 / (50 + 1)
                                    e12 = e50 = None
                                    for i, c in enumerate(candles):
                                        pr = closes[i]
                                        e12 = pr if e12 is None else (pr * alpha12 + e12 * (1 - alpha12))
                                        e50 = pr if e50 is None else (pr * alpha50 + e50 * (1 - alpha50))
                                        c["ema12"] = round(e12, 2)
                                        c["ema50"] = round(e50, 2)
                                res[req_sym] = candles
                            else:
                                # Default to 1d charts map from charts table
                                rows = conn.execute("SELECT symbol, data_json FROM charts").fetchall()
                                for r in rows:
                                    res[r[0]] = json.loads(r[1])
                    except Exception:
                        pass
                data = json.dumps(res).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML.encode())

        def log_message(self, fmt, *args):
            pass   # suppress default request logs

    return DashboardHandler


# ─────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────

def start_web_server(state: BotState, port: int = 8080) -> None:
    """
    Start the web dashboard in a daemon background thread.
    Returns immediately — the server runs until the process exits.
    """
    server = HTTPServer(("0.0.0.0", port), _make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
