# ─────────────────────────────────────────────────────────────────
#  rebalance_rotational.py  —  Live Dhan API Bi-Weekly Rebalancer
#
#  Objective:
#    Connect to Dhan Production API and execute our +35.7% Ann.
#    Dual-Momentum Rotational Strategy across Top 5 Blue-Chips:
#      1. SBIN.NS      (State Bank of India — Banking)
#      2. LT.NS        (Larsen & Toubro — Industrial)
#      3. INFY.NS      (Infosys — IT / Tech)
#      4. TCS.NS       (Tata Consultancy Services — Tech Giant)
#      5. BHARTIARTL.NS(Bharti Airtel — Telecom Leader)
#
#  Execution Protocol:
#    - Smart Hybrid Architecture:
#        * Data Layer   : Free NSEClient (yfinance) for 200 SMA & 60-day momentum
#        * Trading Layer: Dhan API (dhanhq) for fund limits, holdings & order routing
#    - Macro Crash Shield: If NIFTY50 < 200 SMA → Alert to move 100% to Cash/LiquidBEES
#    - Bi-Weekly Rebalancing: Rank top 2 momentum leaders and reallocate
#
#  Usage:
#      python rebalance_rotational.py             # Dry-run / Paper analysis
#      python rebalance_rotational.py --execute   # Place LIVE CNC orders via Dhan
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.india.nse_client import NSEClient

load_dotenv()
console = Console()

TOP_5_GROWTH_LEADERS = [
    "SBIN.NS",
    "LT.NS",
    "INFY.NS",
    "TCS.NS",
    "BHARTIARTL.NS",
]

# Map NSE yfinance symbols to Dhan Security IDs & Trading Symbols (NSE_EQ)
DHAN_SECURITY_MAP = {
    "SBIN.NS":       {"id": "3045",  "symbol": "SBIN"},
    "LT.NS":         {"id": "11483", "symbol": "LT"},
    "INFY.NS":       {"id": "1594",  "symbol": "INFY"},
    "TCS.NS":        {"id": "11536", "symbol": "TCS"},
    "BHARTIARTL.NS": {"id": "10604", "symbol": "BHARTIARTL"},
}


@dataclass
class StockScore:
    symbol:       str
    dhan_symbol:  str
    ltp:          float
    ret_60:       float
    rank:         int = 0
    is_top_2:     bool = False


def get_dhan_client():
    client_id = os.environ.get("DHAN_CLIENT_ID")
    token     = os.environ.get("DHAN_ACCESS_TOKEN")
    if not client_id or not token:
        logger.error("DHAN_CLIENT_ID or DHAN_ACCESS_TOKEN missing in .env file!")
        sys.exit(1)

    try:
        from dhanhq import dhanhq, DhanContext
        ctx  = DhanContext(client_id, token)
        dhan = dhanhq(ctx)
        return dhan
    except Exception as e:
        logger.error(f"Failed to initialize Dhan SDK: {e}")
        sys.exit(1)


def analyze_market_regime_and_rankings(client: NSEClient) -> tuple[bool, float, float, list[StockScore]]:
    logger.info("Fetching NIFTY 50 and Top 5 Blue-Chip daily charts for momentum analysis...")
    df_nifty = client.get_ohlcv("NIFTY50", "1d", 300)
    nifty_close = float(df_nifty["close"].iloc[-1])
    nifty_sma200 = float(df_nifty["close"].rolling(200).mean().iloc[-1])
    is_bull_market = nifty_close > nifty_sma200

    scores = []
    for sym in TOP_5_GROWTH_LEADERS:
        df = client.get_ohlcv(sym, "1d", 150)
        close = float(df["close"].iloc[-1])
        # 60-day price return
        ret60 = (close - float(df["close"].iloc[-61])) / float(df["close"].iloc[-61]) * 100.0 if len(df) >= 61 else 0.0
        scores.append(StockScore(
            symbol      = sym,
            dhan_symbol = DHAN_SECURITY_MAP[sym]["symbol"],
            ltp         = round(close, 2),
            ret_60      = round(ret60, 2),
        ))

    # Sort descendants by 60-day momentum return
    scores.sort(key=lambda x: x.ret_60, reverse=True)
    for idx, s in enumerate(scores):
        s.rank = idx + 1
        s.is_top_2 = (idx < 2) and (s.ret_60 > 0) and is_bull_market

    return is_bull_market, nifty_close, nifty_sma200, scores


def fetch_live_holdings_and_funds(dhan) -> tuple[float, dict[str, int]]:
    avail_cash = 0.0
    holdings_map = {}  # dhan_symbol -> totalQty

    try:
        limits = dhan.get_fund_limits()
        if limits.get("status") == "success":
            avail_cash = float(limits.get("data", {}).get("availabelBalance", 0.0))
        else:
            logger.warning(f"Could not fetch fund limits: {limits.get('remarks')}")
    except Exception as e:
        logger.warning(f"Exception fetching fund limits: {e}")

    try:
        h_res = dhan.get_holdings()
        if h_res.get("status") == "success":
            for h in h_res.get("data", []):
                sym = h.get("tradingSymbol")
                qty = int(h.get("totalQty", 0))
                if sym and qty > 0:
                    holdings_map[sym] = qty
        else:
            logger.info("No current CNC holdings found in Dhan account.")
    except Exception as e:
        logger.warning(f"Exception fetching holdings: {e}")

    return avail_cash, holdings_map


def main():
    parser = argparse.ArgumentParser(description="Live Dhan Bi-Weekly Rotational Rebalancer")
    parser.add_argument("--execute", action="store_true", help="Place LIVE CNC Delivery orders via Dhan API")
    parser.add_argument("--capital", type=float, default=0.0, help="Override target allocation capital (if 0, uses live Dhan available balance)")
    args = parser.parse_args()

    console.print("\n[bold cyan]── DHAN LIVE BI-WEEKLY ROTATIONAL REBALANCER (+35.7% Ann. Strategy) ──[/]")
    console.print("[dim]Target: Top 5 High-Beta Growth Blue-Chips | CNC Cash Delivery | 0% Leverage[/]\n")

    nse_client = NSEClient()
    dhan       = get_dhan_client()

    # 1. Analyze Macro Regime & Stock Momentum Rankings
    is_bull, nifty_ltp, nifty_200, scores = analyze_market_regime_and_rankings(nse_client)

    # 2. Fetch Live Account State
    live_cash, holdings_map = fetch_live_holdings_and_funds(dhan)
    target_capital = args.capital if args.capital > 0 else live_cash
    if target_capital <= 0:
        target_capital = 500_000.0  # Fallback paper analysis capital if live balance is 0
        logger.info(f"Live cash balance is ₹0.00. Using paper analysis capital: ₹{target_capital:,.2f}")

    # 3. Print Macro Crash Shield & Rankings
    reg_col = "green" if is_bull else "red"
    console.print(f"  [bold]Macro Regime Shield[/]: NIFTY 50 = ₹{nifty_ltp:,.2f} | 200 SMA = ₹{nifty_200:,.2f} | "
                  f"Status: [{reg_col}]{'✔ BULL MARKET (Invested)' if is_bull else '🚨 BEAR MARKET (Move 100% to Cash/LiquidBEES)'}[/]\n")

    tbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
    tbl.add_column("Rank",  justify="center", width=6)
    tbl.add_column("Symbol", width=16)
    tbl.add_column("Dhan Sym", width=12)
    tbl.add_column("LTP (₹)", justify="right", width=12)
    tbl.add_column("60-Day Return", justify="right", width=14)
    tbl.add_column("Allocation Status", width=22)

    top_2_symbols = []
    for s in scores:
        r_col = "green" if s.is_top_2 else "dim"
        status_msg = "[bold green]✔ SELECT (Top 2 Leader)[/]" if s.is_top_2 else "[dim]Ignore / Rotate Out[/]"
        if not is_bull:
            status_msg = "[red]🚨 BEAR MARKET (Cash Only)[/]"
        if s.is_top_2:
            top_2_symbols.append(s)

        tbl.add_row(
            f"#{s.rank}",
            f"[{r_col}]{s.symbol}[/]",
            s.dhan_symbol,
            f"₹{s.ltp:>8,.2f}",
            f"{s.ret_60:>7.2f}%",
            status_msg,
        )
    console.print(tbl)

    # 4. Generate Rebalancing Action Plan
    console.print(f"\n  [bold]── Rebalancing Action Plan (Target Capital: ₹{target_capital:,.2f}) ──[/]")
    actions = []

    # 4a. Check for SELL orders (Rotation out of old leaders or bear market liquidation)
    for held_sym, held_qty in holdings_map.items():
        is_still_top2 = any(s.dhan_symbol == held_sym for s in top_2_symbols)
        if not is_bull or not is_still_top2:
            sec_id = next((v["id"] for k, v in DHAN_SECURITY_MAP.items() if v["symbol"] == held_sym), None)
            actions.append({"action": "SELL", "symbol": held_sym, "sec_id": sec_id, "qty": held_qty, "reason": "Rotated out of Top 2" if is_bull else "Bear Market Liquidation"})

    # 4b. Check for BUY orders (Allocation into Top 2 leaders)
    if is_bull and top_2_symbols:
        alloc_per_stock = target_capital / len(top_2_symbols)
        for s in top_2_symbols:
            current_qty = holdings_map.get(s.dhan_symbol, 0)
            target_qty  = int(alloc_per_stock // s.ltp)
            qty_diff    = target_qty - current_qty
            if qty_diff > 0:
                sec_id = DHAN_SECURITY_MAP[s.symbol]["id"]
                actions.append({"action": "BUY", "symbol": s.dhan_symbol, "sec_id": sec_id, "qty": qty_diff, "price": s.ltp, "reason": f"Rank #{s.rank} Leader (+{s.ret_60}% Mom)"})

    if not actions:
        console.print("  [green]✔ Portfolio is actively balanced and aligned with Top 2 leaders! No trades needed today.[/]\n")
        return

    act_table = Table(box=box.DOUBLE_EDGE, show_header=True, header_style="bold cyan")
    act_table.add_column("Action",   width=8)
    act_table.add_column("Stock",    width=14)
    act_table.add_column("Security ID", width=12)
    act_table.add_column("Quantity", justify="right", width=10)
    act_table.add_column("Est. Value (₹)", justify="right", width=14)
    act_table.add_column("Reason / Signal", width=26)

    for a in actions:
        a_col = "green" if a["action"] == "BUY" else "red"
        val   = (a["qty"] * a["price"]) if "price" in a else 0.0
        act_table.add_row(
            f"[bold {a_col}]{a['action']}[/]",
            f"[bold]{a['symbol']}[/]",
            str(a["sec_id"]),
            str(a["qty"]),
            f"₹{val:>10,.2f}" if val > 0 else "Market Order",
            a["reason"],
        )
    console.print(act_table)

    # 5. Order Execution (if --execute flag passed)
    if not args.execute:
        console.print("\n  [yellow]⚠ DRY-RUN / PAPER MODE: To place these LIVE CNC Delivery orders via Dhan API, run:[/]")
        console.print("      [bold cyan]python rebalance_rotational.py --execute[/]\n")
        return

    console.print("\n  [bold red]🚨 PLACING LIVE ORDERS VIA DHAN API IN 3 SECONDS...[/]")
    import time
    time.sleep(3)

    for a in actions:
        if not a["sec_id"]:
            logger.error(f"Cannot execute {a['action']} for {a['symbol']}: Security ID unknown!")
            continue
        try:
            logger.info(f"Submitting LIVE {a['action']} order for {a['qty']} shares of {a['symbol']} (CNC Delivery)...")
            from dhanhq import dhanhq
            order_side = dhanhq.BUY if a["action"] == "BUY" else dhanhq.SELL
            res = dhan.place_order(
                security_id      = a["sec_id"],
                exchange_segment = dhanhq.NSE,
                transaction_type = dhanhq.CNC,
                quantity         = a["qty"],
                order_type       = dhanhq.MARKET,
                product_type     = dhanhq.CNC,
                price            = 0,
            )
            logger.success(f"Order Response [{a['symbol']}]: {res}")
        except Exception as e:
            logger.error(f"Failed to place order for {a['symbol']}: {e}")

    console.print("\n[bold green]✔ Rebalancing execution cycle completed![/]\n")


if __name__ == "__main__":
    main()
