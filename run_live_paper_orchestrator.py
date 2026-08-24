# ─────────────────────────────────────────────────────────────────
#  run_live_paper_orchestrator.py
#  Production-Grade Fault-Tolerant Live Paper Trading Command Center.
#
#  Features:
#    • ACID SQLite Persistence (Crash-Proof, Zero State Loss)
#    • Offline Catch-Up Replay Engine (Replays missed days if PC was off)
#    • Real-Time NSE Data Fetching & Dynamic Regime Meta-Orchestration
#    • Exact Indian CNC Delivery Taxes (STT 0.10%, GST 18%, Stamp, DP)
#    • 3-Tier Risk Hierarchy (1.25 ATR SL, Break-Even Lock, 200 SMA Shield)
#    • Rich Live Dashboard & Daily 3:15 PM Scheduled Daemon
#
#  Usage:
#    python run_live_paper_orchestrator.py --run      # Run live cycle now (10s)
#    python run_live_paper_orchestrator.py --status   # View live portfolio status
#    python run_live_paper_orchestrator.py --reset    # Reset account to ₹1,00,000
#    python run_live_paper_orchestrator.py --daemon   # Run automated daily loop
# ─────────────────────────────────────────────────────────────────

import argparse
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.paper_orchestrator_db import PaperOrchestratorDB
from engine.india_costs import calculate_round_trip_cost
from engine.notifier import Notifier
from indicators import add_all_indicators

console = Console()
notifier = Notifier()
IST = ZoneInfo("Asia/Kolkata")

BENCHMARK_SYMBOL = "^NSEI"
SAFE_ASSET_SYMBOL = "GOLDBEES.NS"
ETFS_ALL = ["NIFTYBEES.NS", "BANKBEES.NS", "CPSEETF.NS"]
STOCKS_ALL = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "LT.NS",
    "BHARTIARTL.NS", "SBIN.NS", "SUNPHARMA.NS", "NTPC.NS"
]
MAX_OPEN_POSITIONS = 3


def fetch_live_market_data() -> Dict[str, pd.DataFrame]:
    """Fetches daily OHLCV bars for all tracked assets from NSE."""
    all_syms = list(set([BENCHMARK_SYMBOL, SAFE_ASSET_SYMBOL] + ETFS_ALL + STOCKS_ALL))
    console.print(f"[dim cyan]Fetching live market data for {len(all_syms)} instruments from NSE...[/]")
    df_raw = yf.download(all_syms, period="1y", interval="1d", progress=False)

    df_closes = df_raw["Close"].copy() if "Close" in df_raw.columns else df_raw.copy()
    if isinstance(df_closes.columns, pd.MultiIndex):
        df_closes.columns = df_closes.columns.get_level_values(0)
    df_closes = df_closes.ffill().dropna(how="all")

    data_map = {}
    for sym in all_syms:
        sub = pd.DataFrame(index=df_closes.index)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            val = df_raw[col][sym] if isinstance(df_raw.columns, pd.MultiIndex) else df_raw[col]
            sub[col.lower()] = val.astype(float)
        sub = sub.ffill().dropna()
        if len(sub) > 30:
            data_map[sym] = add_all_indicators(sub)

    return data_map


def evaluate_macro_regime(df_nifty: pd.DataFrame) -> Dict[str, Any]:
    """Detects active market regime using 200 SMA, ADX(14), and EMA12/50 spread."""
    px = float(df_nifty["close"].iloc[-1])
    sma200 = float(df_nifty["close"].rolling(200).mean().iloc[-1]) if len(df_nifty) >= 200 else px
    adx = float(df_nifty.get("adx", pd.Series([20.0])).iloc[-1])
    ema12 = float(df_nifty.get("ema_12", pd.Series([px])).iloc[-1])
    ema50 = float(df_nifty.get("ema_50", pd.Series([px])).iloc[-1])

    if px <= sma200 or (ema12 < ema50 * 0.99):
        regime = "BEAR_DEFENSE"
        desc = "NIFTY trading below 200 SMA / EMA breakdown. 100% Capital allocated to GOLDBEES / Cash Shield."
    elif adx >= 22.0 and ema12 > ema50:
        regime = "TRENDING_BULL"
        desc = "Strong Bullish Momentum (ADX >= 22). Capital allocated to Sector ETF Momentum & VCP Breakouts."
    else:
        regime = "CHOPPY_SIDEWAYS"
        desc = "Consolidation / Sideways Chop (ADX < 22). Capital allocated to Large-Cap RS Pullbacks & SMC Discount."

    return {
        "regime": regime,
        "nifty_price": px,
        "nifty_sma200": sma200,
        "adx": adx,
        "description": desc,
    }


def execute_offline_catchup(db: PaperOrchestratorDB, data_map: Dict[str, pd.DataFrame]):
    """Replays missed trading days if the local machine was turned off."""
    state = db.get_portfolio_state()
    last_run_str = state.get("last_run_date", "")[:10]
    today_str = datetime.now(IST).strftime("%Y-%m-%d")

    if not last_run_str or last_run_str >= today_str:
        return

    console.print(f"[bold yellow]🔄 Running Offline Catch-Up Replay (Missed period: {last_run_str} ➔ {today_str})...[/]")
    open_pos = db.get_open_positions()

    if not open_pos or BENCHMARK_SYMBOL not in data_map:
        return

    df_bm = data_map[BENCHMARK_SYMBOL]
    missed_dates = [d for d in df_bm.index if d.strftime("%Y-%m-%d") > last_run_str and d.strftime("%Y-%m-%d") < today_str]

    for d in missed_dates:
        d_str = d.strftime("%Y-%m-%d")
        for pos in list(open_pos):
            sym = pos["symbol"]
            df_s = data_map.get(sym)
            if df_s is None or d not in df_s.index:
                continue

            bar = df_s.loc[d]
            low_px = float(bar["low"])
            high_px = float(bar["high"])
            close_px = float(bar["close"])

            # Check SL Trigger on missed day
            if low_px <= pos["stop_loss"]:
                b_c, s_c = calculate_round_trip_cost(pos["entry_price"], pos["stop_loss"], pos["quantity"], "CNC")
                tax = b_c.total + s_c.total
                gross_sale = pos["stop_loss"] * pos["quantity"]
                net_gain = (pos["stop_loss"] - pos["entry_price"]) * pos["quantity"] - tax

                db.close_position(sym, pos["stop_loss"], d_str, "STOP_LOSS (CATCHUP)", tax)
                curr_cash = state["cash_balance_inr"] + gross_sale - s_c.total
                curr_realized = state["realized_pnl_inr"] + net_gain
                curr_tax = state["total_taxes_paid_inr"] + tax
                db.update_portfolio_state(curr_cash, 0.0, curr_cash, curr_realized, curr_tax, state["active_regime"], d_str)
                console.print(f"[bold red]🛑 Catch-Up Exit: {sym} hit Stop Loss at ₹{pos['stop_loss']:.2f} on {d_str}[/]")

            # Check TP Trigger on missed day
            elif high_px >= pos["take_profit"]:
                b_c, s_c = calculate_round_trip_cost(pos["entry_price"], pos["take_profit"], pos["quantity"], "CNC")
                tax = b_c.total + s_c.total
                gross_sale = pos["take_profit"] * pos["quantity"]
                net_gain = (pos["take_profit"] - pos["entry_price"]) * pos["quantity"] - tax

                db.close_position(sym, pos["take_profit"], d_str, "TAKE_PROFIT (CATCHUP)", tax)
                curr_cash = state["cash_balance_inr"] + gross_sale - s_c.total
                curr_realized = state["realized_pnl_inr"] + net_gain
                curr_tax = state["total_taxes_paid_inr"] + tax
                db.update_portfolio_state(curr_cash, 0.0, curr_cash, curr_realized, curr_tax, state["active_regime"], d_str)
                console.print(f"[bold green]🎯 Catch-Up Exit: {sym} hit Take Profit at ₹{pos['take_profit']:.2f} on {d_str}[/]")


def run_live_cycle(db: PaperOrchestratorDB, force: bool = False, is_cron: bool = False):
    """Executes the complete live paper trading cycle in 10 seconds with 0.01s idempotency check."""
    now = datetime.now(IST)
    today_str = now.strftime("%Y-%m-%d")
    state = db.get_portfolio_state()
    last_run_str = state.get("last_run_date", "")[:10]

    # Fast Idempotency Guards for Cron Spawns (0.01s exit, 0% CPU)
    if is_cron and not force:
        # If already completed today, exit immediately with zero CPU
        if last_run_str == today_str:
            console.print(f"[dim yellow]⏩ [Fast Skip] Today's execution ({today_str}) has already completed. Exiting in 0.01s.[/]")
            return

        # If weekday before 15:10 IST, skip until near market close
        if now.weekday() < 5 and (now.hour < 15 or (now.hour == 15 and now.minute < 10)):
            console.print(f"[dim cyan]⏳ [Market Open] Current time is {now.strftime('%H:%M IST')}. Daily scan window triggers after 15:10 IST. Exiting in 0.01s.[/]")
            return

    console.print(f"[bold cyan]🚀 Starting Live Paper Execution Cycle for {today_str} ({now.strftime('%H:%M:%S IST')})...[/]")
    data_map = fetch_live_market_data()

    if BENCHMARK_SYMBOL not in data_map:
        console.print("[bold red]❌ Error: Unable to fetch benchmark market data from NSE.[/]")
        return

    # 1. Run Offline Catch-Up Replay if PC was off
    execute_offline_catchup(db, data_map)

    # 2. Evaluate Active Macro Regime
    df_bm = data_map[BENCHMARK_SYMBOL]
    regime_info = evaluate_macro_regime(df_bm)
    regime = regime_info["regime"]
    bm_px = regime_info["nifty_price"]

    state = db.get_portfolio_state()
    cash = state["cash_balance_inr"]
    realized_pnl = state["realized_pnl_inr"]
    total_taxes = state["total_taxes_paid_inr"]

    open_pos = db.get_open_positions()

    # 3. Monitor & Exit Open Positions
    invested = 0.0
    for pos in open_pos:
        sym = pos["symbol"]
        df_s = data_map.get(sym)
        if df_s is None: continue

        curr_px = float(df_s["close"].iloc[-1])
        entry_px = pos["entry_price"]
        sl_px = pos["stop_loss"]
        tp_px = pos["take_profit"]
        qty = pos["quantity"]
        risk_unit = pos["risk_unit"]

        # Check Break-Even Trailing Lock (+2.00 ATR Gain)
        be_locked = bool(pos.get("be_locked", 0))
        if not be_locked and curr_px >= (entry_px + 2.00 * risk_unit):
            sl_px = entry_px
            be_locked = True
            console.print(f"[bold cyan]🛡️ Trailing Stop Activated: {sym} reached +2.0R gain! SL moved to Entry (₹{entry_px:.2f})[/]")

        # Check Time Exit (45 trading days ≈ 65 calendar days)
        days_held = 0
        if pos.get("entry_date"):
            try:
                days_held = (now.date() - datetime.strptime(pos["entry_date"][:10], "%Y-%m-%d").date()).days
            except Exception:
                pass

        # Check SL Exit
        if curr_px <= sl_px:
            b_c, s_c = calculate_round_trip_cost(entry_px, curr_px, qty, "CNC")
            tax = b_c.total + s_c.total
            gross_sale = curr_px * qty
            net_gain = (curr_px - entry_px) * qty - tax

            db.close_position(sym, curr_px, today_str, "STOP_LOSS", tax)
            cash += gross_sale - s_c.total
            realized_pnl += net_gain
            total_taxes += tax
            console.print(f"[bold red]🛑 Position Closed (SL Hit): {sym} sold at ₹{curr_px:,.2f} | Net P&L: ₹{net_gain:,.2f}[/]")
            notifier.notify_trade_exit(sym, pos["strategy_name"], curr_px, net_gain, (net_gain / (entry_px * qty)) * 100.0 if (entry_px * qty) > 0 else 0.0, "STOP_LOSS", tax)

        # Check TP Exit
        elif curr_px >= tp_px:
            b_c, s_c = calculate_round_trip_cost(entry_px, curr_px, qty, "CNC")
            tax = b_c.total + s_c.total
            gross_sale = curr_px * qty
            net_gain = (curr_px - entry_px) * qty - tax

            db.close_position(sym, curr_px, today_str, "TAKE_PROFIT", tax)
            cash += gross_sale - s_c.total
            realized_pnl += net_gain
            total_taxes += tax
            console.print(f"[bold green]🎯 Position Closed (TP Hit): {sym} sold at ₹{curr_px:,.2f} | Net P&L: +₹{net_gain:,.2f}[/]")
            notifier.notify_trade_exit(sym, pos["strategy_name"], curr_px, net_gain, (net_gain / (entry_px * qty)) * 100.0 if (entry_px * qty) > 0 else 0.0, "TAKE_PROFIT", tax)

        # Check Bear Market Rotation Exit (NIFTY < 200 SMA)
        elif regime == "BEAR_DEFENSE" and sym != SAFE_ASSET_SYMBOL:
            b_c, s_c = calculate_round_trip_cost(entry_px, curr_px, qty, "CNC")
            tax = b_c.total + s_c.total
            gross_sale = curr_px * qty
            net_gain = (curr_px - entry_px) * qty - tax

            db.close_position(sym, curr_px, today_str, "BEAR_DEFENSE_ROTATION", tax)
            cash += gross_sale - s_c.total
            realized_pnl += net_gain
            total_taxes += tax
            console.print(f"[bold yellow]🛡️ Bear Defense Exit: Liquidated {sym} at ₹{curr_px:,.2f} to protect capital.[/]")
            notifier.notify_trade_exit(sym, pos["strategy_name"], curr_px, net_gain, (net_gain / (entry_px * qty)) * 100.0 if (entry_px * qty) > 0 else 0.0, "BEAR_DEFENSE_ROTATION", tax)

        # Check Time-Based Forced Exit (45 trading days)
        elif days_held >= 65 and sym != SAFE_ASSET_SYMBOL:
            b_c, s_c = calculate_round_trip_cost(entry_px, curr_px, qty, "CNC")
            tax = b_c.total + s_c.total
            gross_sale = curr_px * qty
            net_gain = (curr_px - entry_px) * qty - tax

            db.close_position(sym, curr_px, today_str, "TIME_EXIT", tax)
            cash += gross_sale - s_c.total
            realized_pnl += net_gain
            total_taxes += tax
            console.print(f"[bold yellow]⏰ Time Exit (45d Stagnation): Liquidated {sym} at ₹{curr_px:,.2f} | Net P&L: ₹{net_gain:,.2f}[/]")
            notifier.notify_trade_exit(sym, pos["strategy_name"], curr_px, net_gain, (net_gain / (entry_px * qty)) * 100.0 if (entry_px * qty) > 0 else 0.0, "TIME_EXIT", tax)

        else:
            db.update_position_price(sym, curr_px, be_locked=be_locked, stop_loss=sl_px)
            invested += curr_px * qty

    # 4. Scan & Enter New Paper Positions
    open_pos = db.get_open_positions()
    available_slots = MAX_OPEN_POSITIONS - len(open_pos)

    if regime == "BEAR_DEFENSE":
        df_gold = data_map.get(SAFE_ASSET_SYMBOL)
        if df_gold is not None and len(df_gold) >= 50:
            g_px = float(df_gold["close"].iloc[-1])
            g_ema50 = float(df_gold["close"].ewm(span=50, adjust=False).mean().iloc[-1])
            g_sma200 = float(df_gold["close"].rolling(200).mean().iloc[-1]) if len(df_gold) >= 200 else g_px

            # Gold Dual-Trend Macro Gate: Only buy Gold if Gold > 50-EMA AND Gold > 200-SMA
            if g_px > g_ema50 and g_px > g_sma200:
                if SAFE_ASSET_SYMBOL not in [p["symbol"] for p in open_pos] and cash > 2000:
                    g_qty = math.floor((cash * 0.95) / g_px)
                    if g_qty > 0:
                        b_c, _ = calculate_round_trip_cost(g_px, g_px, g_qty, "CNC")
                        invest_cost = g_qty * g_px + b_c.total
                        if cash >= invest_cost:
                            cash -= invest_cost
                            total_taxes += b_c.total
                            invested += g_qty * g_px
                            db.add_position({
                                "symbol": SAFE_ASSET_SYMBOL,
                                "strategy_name": "Sovereign Gold Defense Shield",
                                "entry_date": today_str,
                                "entry_price": g_px,
                                "quantity": g_qty,
                                "current_price": g_px,
                                "stop_loss": round(g_px * 0.92, 2),    # 8.0% SL
                                "take_profit": round(g_px * 1.15, 2),  # 15.0% TP
                                "risk_unit": round(g_px * 0.08, 2),
                                "be_locked": False,
                            })
                            console.print(f"[bold yellow]🛡️ Entered Sovereign Gold Defense: Bought {g_qty} units of GOLDBEES at ₹{g_px:.2f} (SL: ₹{g_px*0.92:.2f}, TP: ₹{g_px*1.15:.2f})[/]")
                            notifier.notify_trade_entry(SAFE_ASSET_SYMBOL, "Sovereign Gold Defense Shield", g_px, g_qty, round(g_px * 0.92, 2), round(g_px * 1.15, 2), 1.88, regime)
            else:
                console.print(f"[bold cyan]🛡️ Gold Dual-Trend Gate: GOLDBEES (₹{g_px:.2f}) is below 50 EMA (₹{g_ema50:.2f}) or 200 SMA (₹{g_sma200:.2f}). Holding 100% Cash Shield to prevent drawdown.[/]")

    elif available_slots > 0 and cash > 5000:
        alloc_per_slot = cash / available_slots

        # Scan Large-Cap RS Pullbacks & Breakouts
        candidates = []
        for sym in STOCKS_ALL:
            if sym in [p["symbol"] for p in open_pos]: continue
            df_s = data_map.get(sym)
            if df_s is None or len(df_s) < 60: continue

            px = float(df_s["close"].iloc[-1])
            op = float(df_s["open"].iloc[-1]) if "open" in df_s else px
            hi = float(df_s["high"].iloc[-1]) if "high" in df_s else px
            lo = float(df_s["low"].iloc[-1]) if "low" in df_s else px
            prev_px = float(df_s["close"].iloc[-2])
            sma20 = float(df_s["sma_20"].iloc[-1]) if "sma_20" in df_s else px
            prev_sma20 = float(df_s["sma_20"].iloc[-2]) if "sma_20" in df_s else prev_px
            rsi = float(df_s.get("rsi", pd.Series([50.0])).iloc[-1])
            atr = float(df_s.get("atr", pd.Series([px * 0.02])).iloc[-1])

            # 60d RS vs NIFTY
            rs_today = px / bm_px
            rs_60 = float(df_s["close"].iloc[-60]) / float(df_bm["close"].iloc[-60])
            rs_slope = ((rs_today - rs_60) / rs_60) * 100.0

            # Bullish Green Reversal Filter (Loss Minimizer: Green Bar + Upper 50% Close)
            is_bullish_reversal = (px >= op) and (hi > lo and px >= (lo + 0.50 * (hi - lo)))

            # Pullback Setup: Wholesale Dip at SMA20 + Healthy RSI + Positive RS Slope + Bullish Green Reversal
            is_pullback = (prev_px <= prev_sma20 * 1.008) and (px > sma20) and (40.0 <= rsi <= 60.0) and (rs_slope > 0) and is_bullish_reversal

            if is_pullback:
                candidates.append((sym, "Large-Cap RS Pullback", px, atr, rs_slope))

        candidates.sort(key=lambda x: x[4], reverse=True)  # Sort by highest RS slope

        for sym, strat, px, atr, rs in candidates[:available_slots]:
            qty = math.floor(alloc_per_slot / (px * 1.005))
            if qty > 0:
                b_c, _ = calculate_round_trip_cost(px, px, qty, "CNC")
                total_order_cost = (qty * px) + b_c.total
                if cash >= total_order_cost:
                    sl = round(px - 1.25 * atr, 2)
                    tp = round(px + 4.00 * atr, 2)
                    cash -= total_order_cost
                    total_taxes += b_c.total
                    invested += qty * px

                    db.add_position({
                        "symbol": sym,
                        "strategy_name": strat,
                        "entry_date": today_str,
                        "entry_price": px,
                        "quantity": qty,
                        "current_price": px,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "risk_unit": round(1.25 * atr, 2),
                        "be_locked": False,
                    })
                    console.print(f"[bold green]🚀 Paper Buy Executed: {qty} shs of {sym} at ₹{px:,.2f} (SL: ₹{sl:,.2f} | TP: ₹{tp:,.2f})[/]")
                    notifier.notify_trade_entry(sym, strat, px, qty, sl, tp, 3.2, regime)

    # 5. Update SQLite State
    current_nav = cash + invested
    db.update_portfolio_state(cash, invested, current_nav, realized_pnl, total_taxes, regime, today_str)
    db.log_daily_nav(today_str, current_nav, cash, invested, bm_px, regime)
    notifier.notify_daily_summary(current_nav, cash, invested, realized_pnl, len(db.get_open_positions()), regime)

    # 6. Render Dashboard
    display_status(db, regime_info)


def display_status(db: PaperOrchestratorDB, regime_info: Optional[Dict[str, Any]] = None):
    """Renders the live Rich portfolio dashboard."""
    state = db.get_portfolio_state()
    positions = db.get_open_positions()
    history = db.get_trade_history(limit=5)

    cash = state["cash_balance_inr"]
    invested = sum(p["current_price"] * p["quantity"] for p in positions)
    nav = cash + invested
    peak = state["peak_nav_inr"]
    realized = state["realized_pnl_inr"]
    taxes = state["total_taxes_paid_inr"]
    regime = state["active_regime"]
    dd = ((peak - nav) / peak * 100.0) if peak > 0 else 0.0

    console.print()
    console.print(Panel.fit(
        f"[bold cyan]🏛️ AI MULTI-STRATEGY META-ORCHESTRATOR: LIVE PAPER TRADING CENTER[/]\n"
        f"[bold]Active Macro Regime :[/] [bold {'green' if regime=='TRENDING_BULL' else ('yellow' if regime=='CHOPPY_SIDEWAYS' else 'red')}]{regime}[/]\n"
        f"[dim]Last Updated: {state['updated_at']} IST[/]",
        border_style="cyan"
    ))
    console.print()

    # Portfolio Summary Table
    tbl_sum = Table(title="[bold green]💼 PORTFOLIO FINANCIAL OVERVIEW[/]", box=box.ROUNDED, header_style="bold cyan")
    tbl_sum.add_column("Current NAV Value", justify="right", style="bold green", width=22)
    tbl_sum.add_column("Available Cash", justify="right", style="bold cyan", width=20)
    tbl_sum.add_column("Invested Equities", justify="right", width=20)
    tbl_sum.add_column("Realized Net P&L", justify="right", width=20)
    tbl_sum.add_column("Total Taxes Paid", justify="right", style="dim", width=18)
    tbl_sum.add_column("Peak NAV DD", justify="right", width=14)

    tbl_sum.add_row(
        f"₹{nav:,.2f}",
        f"₹{cash:,.2f}",
        f"₹{invested:,.2f}",
        f"[bold green]+₹{realized:,.2f}[/]" if realized >= 0 else f"[bold red]-₹{abs(realized):,.2f}[/]",
        f"₹{taxes:,.2f}",
        f"-{dd:.2f}%",
    )
    console.print(tbl_sum)
    console.print()

    # Open Positions Table
    tbl_pos = Table(title="[bold green]🎯 CURRENT OPEN POSITIONS (CNC DELIVERY)[/]", box=box.ROUNDED, header_style="bold cyan")
    tbl_pos.add_column("Symbol", style="bold", width=14)
    tbl_pos.add_column("Strategy Engine", width=26)
    tbl_pos.add_column("Entry Date", justify="center", width=12)
    tbl_pos.add_column("Entry Px", justify="right", width=12)
    tbl_pos.add_column("Live Px", justify="right", width=12)
    tbl_pos.add_column("Qty", justify="right", width=8)
    tbl_pos.add_column("Stop Loss", justify="right", style="bold red", width=12)
    tbl_pos.add_column("Target (TP)", justify="right", style="bold green", width=12)
    tbl_pos.add_column("Unrealized P&L", justify="right", width=16)

    if positions:
        for p in positions:
            unreal = (p["current_price"] - p["entry_price"]) * p["quantity"]
            unreal_pct = ((p["current_price"] - p["entry_price"]) / p["entry_price"]) * 100.0 if p["entry_price"] > 0 else 0.0
            pnl_str = f"[bold green]+₹{unreal:,.2f} (+{unreal_pct:.1f}%)[/]" if unreal >= 0 else f"[bold red]-₹{abs(unreal):,.2f} ({unreal_pct:.1f}%)[/]"
            tbl_pos.add_row(
                p["symbol"], p["strategy_name"], p["entry_date"][:10],
                f"₹{p['entry_price']:,.2f}", f"₹{p['current_price']:,.2f}", str(p["quantity"]),
                f"₹{p['stop_loss']:,.2f}", f"₹{p['take_profit']:,.2f}", pnl_str
            )
    else:
        tbl_pos.add_row("—", "No active open positions (100% Cash / Awaiting Setups)", "—", "—", "—", "—", "—", "—", "—")

    console.print(tbl_pos)
    console.print()

    # Recent Closed Trades Table
    if history:
        tbl_hist = Table(title="[bold yellow]📜 RECENT COMPLETED TRADES (CLOSED)[/]", box=box.ROUNDED, header_style="bold yellow")
        tbl_hist.add_column("Symbol", style="bold", width=14)
        tbl_hist.add_column("Strategy", width=24)
        tbl_hist.add_column("Exit Date", justify="center", width=12)
        tbl_hist.add_column("Entry Px", justify="right", width=12)
        tbl_hist.add_column("Exit Px", justify="right", width=12)
        tbl_hist.add_column("Qty", justify="right", width=8)
        tbl_hist.add_column("Net P&L (₹)", justify="right", width=16)
        tbl_hist.add_column("Taxes Paid", justify="right", style="dim", width=12)
        tbl_hist.add_column("Exit Reason", justify="center", width=22)

        for h in history:
            net_pnl = h["net_pnl"]
            pnl_str = f"[bold green]+₹{net_pnl:,.2f} (+{h['net_pnl_pct']:.1f}%)[/]" if net_pnl >= 0 else f"[bold red]-₹{abs(net_pnl):,.2f} ({h['net_pnl_pct']:.1f}%)[/]"
            tbl_hist.add_row(
                h["symbol"], h["strategy_name"], h["exit_date"][:10],
                f"₹{h['entry_price']:,.2f}", f"₹{h['exit_price']:,.2f}", str(h["quantity"]),
                pnl_str, f"₹{h['taxes_inr']:,.2f}", h["exit_reason"]
            )
        console.print(tbl_hist)
        console.print()


def run_daemon_loop(db: PaperOrchestratorDB):
    """Runs a background loop that executes the daily scan at 3:15 PM IST."""
    console.print(Panel.fit(
        "[bold cyan]🔄 LIVE PAPER TRADING DAEMON ACTIVATED[/]\n"
        "[dim]Schedules execution daily at 3:15 PM IST. Resilient to sleep/wake cycles.[/]",
        border_style="cyan"
    ))

    last_executed_day = None

    while True:
        now = datetime.now(IST)
        today_str = now.strftime("%Y-%m-%d")
        weekday = now.weekday()  # 0 = Monday, 4 = Friday

        # Check if today is a weekday and time is past 15:10 IST
        is_market_day = (weekday < 5)
        is_scan_time = (now.hour == 15 and now.minute >= 10) or (now.hour > 15)

        if is_market_day and is_scan_time and last_executed_day != today_str:
            console.print(f"[bold green]⏰ Triggering Daily 3:15 PM Market Execution ({now.strftime('%H:%M:%S IST')})...[/]")
            try:
                run_live_cycle(db)
                last_executed_day = today_str
            except Exception as e:
                console.print(f"[bold red]❌ Error executing daily cycle: {e}[/]")

        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description="AI Quantitative Multi-Strategy Paper Trading Command Center")
    parser.add_argument("--run", action="store_true", help="Execute live market scan and update paper portfolio now (10s)")
    parser.add_argument("--cron", action="store_true", help="Fast idempotency check mode for frequent cron spawns (0.01s exit if already done)")
    parser.add_argument("--force", action="store_true", help="Force execute live scan regardless of whether today already completed")
    parser.add_argument("--status", action="store_true", help="View current live portfolio NAV, positions and trade logs")
    parser.add_argument("--reset", action="store_true", help="Reset paper trading portfolio to fresh ₹1,00,000 capital")
    parser.add_argument("--daemon", action="store_true", help="Run in background daemon mode (auto-executes at 3:15 PM IST)")
    args = parser.parse_args()

    db = PaperOrchestratorDB()

    if args.reset:
        db.reset_portfolio(100000.0)
        console.print("[bold green]✅ Paper trading portfolio reset to fresh ₹1,00,000.00 capital.[/]")
        display_status(db)
    elif args.status:
        display_status(db)
    elif args.daemon:
        run_daemon_loop(db)
    elif args.cron:
        run_live_cycle(db, force=args.force, is_cron=True)
    else:
        # Default to --run (interactive or forced)
        run_live_cycle(db, force=args.force, is_cron=False)


if __name__ == "__main__":
    main()
