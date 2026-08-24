# ─────────────────────────────────────────────────────────────────
#  india_main.py  —  India Market Paper Trading Bot (NSE/BSE)
#
#  Usage:
#      source .venv/bin/activate && python india_main.py
#      python india_main.py --symbol BANKNIFTY --interval 5m
#
#  Web Dashboard:
#      http://localhost:8081  (separate port from BTC bot)
#
#  Stop:
#      Ctrl+C  →  saves ledger and exits cleanly
#
#  Loop behaviour (every PRICE_CHECK_INTERVAL seconds):
#    1. Check IST session window + NSE holiday calendar → skip if closed
#    2. Refresh India VIX (once per scan, not every loop)
#    3. Fetch candles + compute indicators
#    4. Detect regime (ADX + BB width + VIX)
#    5. Score with IndiaSignalScorer (IST gate, VIX multiplier)
#    6. Open / close positions via broker (paper by default)
#    7. Check open positions against SL / TP
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from zoneinfo  import ZoneInfo

from loguru import logger
from rich.console import Console
from rich.live    import Live
from rich.panel   import Panel
from rich.table   import Table
from rich import box

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.india_settings import (
    INDIA_DEFAULT_SYMBOL, INDIA_DEFAULT_INTERVAL, INDIA_DEFAULT_BARS,
    INITIAL_CAPITAL_INR, INDIA_SIGNAL_THRESHOLD,
    INDIA_ATR_SL_MULT, INDIA_ATR_TP_MULT,
    INDIA_MAX_RISK_PER_TRADE_PCT, INDIA_MAX_OPEN_TRADES,
    INDIA_DAILY_LOSS_LIMIT_PCT,
)
from data.india.nse_client      import NSEClient
from data.india.india_calendar  import IndiaCalendar
from indicators                 import add_all_indicators
from probability.india_signal_scorer import IndiaSignalScorer
from engine.india_regime        import detect_india_regime
from engine                     import Portfolio, OrderManager, Ledger
from brokers.paper_broker       import PaperBrokerIndia
from engine.india_costs         import effective_cost_pct

IST = ZoneInfo("Asia/Kolkata")

# ─────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────

PRICE_CHECK_INTERVAL = 60    # seconds between loop ticks
VIX_REFRESH_EVERY    = 5     # refresh VIX every N loop ticks
WEB_PORT             = 8081

console = Console()


# ─────────────────────────────────────────────────────────────────
#  Candle boundary helpers
# ─────────────────────────────────────────────────────────────────

_INTERVAL_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "1d": 1440,
}


def secs_until_next_candle(interval: str) -> float:
    """Seconds until the next candle of the given interval closes (UTC)."""
    interval_min = _INTERVAL_MINUTES.get(interval, 15)
    now          = datetime.now(timezone.utc)
    total_secs   = now.hour * 3600 + now.minute * 60 + now.second
    interval_sec = interval_min * 60
    elapsed      = total_secs % interval_sec
    remaining    = interval_sec - elapsed
    return remaining + 5.0   # 5s buffer after close


def candle_just_closed(interval: str) -> bool:
    """True in the window right after a candle of this interval closed."""
    remaining = secs_until_next_candle(interval)
    interval_sec = _INTERVAL_MINUTES.get(interval, 15) * 60
    return remaining > (interval_sec - PRICE_CHECK_INTERVAL - 10)


# ─────────────────────────────────────────────────────────────────
#  Signal scanning
# ─────────────────────────────────────────────────────────────────

def scan_india_signal(
    client:   NSEClient,
    scorer:   IndiaSignalScorer,
    symbol:   str,
    interval: str,
    bars:     int,
):
    """Fetch candles, compute indicators, score. Returns TradeSignal or None."""
    try:
        df = client.get_ohlcv(symbol, interval, bars)
        df = add_all_indicators(df)
        last = df.iloc[-2]   # last fully closed candle
        signal = scorer.score(last)
        logger.info(
            f"[{symbol}] signal={signal.direction} prob={signal.probability:.1f}% "
            f"reason={signal.reason}"
        )
        return signal
    except Exception as e:
        logger.warning(f"Signal scan failed for {symbol}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
#  Position monitoring against broker SL/TP
# ─────────────────────────────────────────────────────────────────

def check_open_positions(
    portfolio: Portfolio,
    broker:    PaperBrokerIndia,
    client:    NSEClient,
) -> None:
    """
    Check each open position against its SL and TP.
    Uses AIAO model: close entire position at first SL or TP hit.
    """
    if not portfolio.positions:
        return

    prices = {}
    for trade_id, pos in list(portfolio.positions.items()):
        sym = pos.symbol
        if sym not in prices:
            try:
                prices[sym] = client.get_price(sym)
            except Exception:
                continue

        price = prices[sym]
        hit_sl = price <= pos.stop_loss if pos.direction == "LONG" else price >= pos.stop_loss
        hit_tp = price >= pos.take_profit1 if pos.direction == "LONG" else price <= pos.take_profit1

        if hit_sl or hit_tp:
            reason = "SL" if hit_sl else "TP"
            result = broker.place_market_order(sym, "SELL", pos.quantity)
            if result.success:
                logger.success(
                    f"[{sym}] {reason} hit at ₹{price:,.2f} — "
                    f"closed {pos.quantity} units (filled ₹{result.price:,.2f})"
                )
            else:
                logger.error(f"[{sym}] Close order failed: {result.message}")


# ─────────────────────────────────────────────────────────────────
#  Rich terminal display
# ─────────────────────────────────────────────────────────────────

def build_india_layout(
    symbol:       str,
    interval:     str,
    broker:       PaperBrokerIndia,
    vix:          float,
    last_signal,
    start_time:   datetime,
    next_candle:  int,
) -> Panel:
    now_ist = datetime.now(tz=IST)
    uptime  = datetime.now(timezone.utc) - start_time

    balance   = broker.get_balance()
    positions = broker.get_positions()
    pnl_open  = sum(p.pnl for p in positions)
    total_val = balance + sum(p.qty * p.current_price for p in positions)

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", expand=True)
    t.add_column("Key",   style="dim", width=22)
    t.add_column("Value", justify="right")

    t.add_row("Symbol / Interval", f"{symbol} / {interval}")
    t.add_row("IST Time",          now_ist.strftime("%H:%M:%S IST"))
    t.add_row("India VIX",         f"{vix:.2f}")
    t.add_row("Balance (INR)",     f"₹{balance:,.2f}")
    t.add_row("Open Positions",    str(len(positions)))
    t.add_row("Unrealised PnL",    f"₹{pnl_open:,.2f}")
    t.add_row("Portfolio Value",   f"₹{total_val:,.2f}")
    t.add_row("Next candle in",    f"{next_candle}s")
    t.add_row("Uptime",            str(uptime).split(".")[0])

    if last_signal:
        sig_color = {"LONG": "green", "SHORT": "red"}.get(last_signal.direction, "white")
        t.add_row(
            "Last Signal",
            f"[{sig_color}]{last_signal.direction}[/] "
            f"{last_signal.probability:.1f}%",
        )

    return Panel(t, title="[bold]India Market Bot[/]  |  NSE Paper Trading", border_style="cyan")


# ─────────────────────────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="India NSE paper trading bot")
    parser.add_argument("--symbol",   default=INDIA_DEFAULT_SYMBOL,   help="e.g. NIFTY50 or BANKNIFTY")
    parser.add_argument("--interval", default=INDIA_DEFAULT_INTERVAL,  help="e.g. 5m 15m 1h")
    parser.add_argument("--bars",     default=INDIA_DEFAULT_BARS, type=int, help="Candles to fetch")
    parser.add_argument("--capital",  default=INITIAL_CAPITAL_INR, type=float, help="Starting capital INR")
    args = parser.parse_args()

    console.print(f"\n[bold cyan]India Market Paper Bot — {args.symbol} {args.interval}[/]\n")

    # ── Bootstrap ─────────────────────────────────────────────────
    client    = NSEClient()
    calendar  = IndiaCalendar()
    broker    = PaperBrokerIndia(initial_balance_inr=args.capital)
    portfolio = Portfolio()
    ledger    = Ledger()
    orders    = OrderManager(portfolio, ledger)

    scorer    = IndiaSignalScorer(
        symbol=args.symbol,
        long_threshold=INDIA_SIGNAL_THRESHOLD,
        current_vix=15.0,
    )

    # ── Initial VIX fetch ──────────────────────────────────────────
    try:
        current_vix = client.get_india_vix()
        logger.info(f"India VIX = {current_vix:.2f}")
    except Exception:
        current_vix = 15.0
        logger.warning("VIX fetch failed — defaulting to 15.0")

    scorer.current_vix = current_vix   # scorer reads this attribute

    start_time    = datetime.now(timezone.utc)
    last_signal   = None
    vix_tick      = 0

    try:
        while True:
            loop_start = time.time()
            now_ist    = datetime.now(tz=IST)

            # ── 1. Session gate ────────────────────────────────────
            if not calendar.is_market_open(now_ist):
                secs_to_open = calendar.seconds_to_open(now_ist)
                wait         = min(secs_to_open, 300)  # max 5min sleep
                next_candle  = int(secs_until_next_candle(args.interval))
                layout = build_india_layout(
                    args.symbol, args.interval, broker,
                    current_vix, last_signal, start_time, next_candle,
                )
                console.print(
                    layout,
                    f"\n[dim]Market closed. Opens in {secs_to_open:.0f}s. "
                    f"Sleeping {wait:.0f}s...[/]"
                )
                time.sleep(wait)
                continue

            # ── 2. VIX refresh (every N ticks) ────────────────────
            vix_tick += 1
            if vix_tick >= VIX_REFRESH_EVERY:
                vix_tick = 0
                try:
                    current_vix = client.get_india_vix()
                    scorer.current_vix = current_vix
                except Exception as e:
                    logger.warning(f"VIX refresh failed: {e}")

            # ── 3. Position monitoring ─────────────────────────────
            check_open_positions(portfolio, broker, client)

            # ── 4. Signal scan on candle close ─────────────────────
            if candle_just_closed(args.interval):
                signal = scan_india_signal(
                    client, scorer, args.symbol, args.interval, args.bars
                )
                if signal:
                    last_signal = signal

                if signal and signal.is_tradeable() and signal.probability >= INDIA_SIGNAL_THRESHOLD:
                    # Compute lot size from risk
                    price     = signal.entry_price
                    sl_dist   = abs(price - signal.stop_loss)
                    risk_inr  = broker.get_balance() * INDIA_MAX_RISK_PER_TRADE_PCT / 100
                    qty       = int(risk_inr / sl_dist) if sl_dist > 0 else 0

                    if qty > 0:
                        result = broker.place_market_order(
                            args.symbol,
                            "BUY" if signal.direction == "LONG" else "SELL",
                            qty,
                        )
                        if result.success:
                            # Register with OrderManager so daily loss limit is tracked
                            orders.try_open(signal)
                            logger.success(
                                f"[{args.symbol}] Opened {qty} units @ ₹{result.price:,.2f} "
                                f"SL=₹{signal.stop_loss:,.2f} TP=₹{signal.take_profit1:,.2f}"
                            )
                        else:
                            logger.warning(f"Order rejected: {result.message}")

            # ── 5. Terminal display ────────────────────────────────
            next_candle = int(secs_until_next_candle(args.interval))
            layout = build_india_layout(
                args.symbol, args.interval, broker,
                current_vix, last_signal, start_time, next_candle,
            )
            console.clear()
            console.print(layout)

            # ── 6. Sleep ───────────────────────────────────────────
            elapsed    = time.time() - loop_start
            sleep_time = max(0, PRICE_CHECK_INTERVAL - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass

    # ── Graceful shutdown ─────────────────────────────────────────
    console.print("\n[bold yellow]Stopping India bot...[/]")

    if ledger._records:
        path = ledger.save()
        console.print(f"[green]Ledger saved: {path}[/]")

    balance   = broker.get_balance()
    positions = broker.get_positions()
    pnl_open  = sum(p.pnl for p in positions)
    total_val = balance + sum(p.qty * p.current_price for p in positions)
    net_pnl   = total_val - args.capital

    console.print(
        f"\n[bold cyan]Final Stats[/]\n"
        f"  Starting capital : ₹{args.capital:,.2f}\n"
        f"  Final portfolio  : ₹{total_val:,.2f}\n"
        f"  Net PnL          : {'[green]' if net_pnl >= 0 else '[red]'}₹{net_pnl:,.2f}[/]\n"
        f"  VIX at exit      : {current_vix:.2f}\n"
    )


if __name__ == "__main__":
    main()
