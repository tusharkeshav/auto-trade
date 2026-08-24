# ─────────────────────────────────────────────────────────────────
#  run_india_backtest.py  —  India NSE backtest runner
#
#  Usage:
#      python run_india_backtest.py
#      python run_india_backtest.py --symbol BANKNIFTY --interval 1h
#      python run_india_backtest.py --symbol NIFTY50 --interval 15m --bars 1400 --threshold 50
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.india.nse_client import NSEClient
from backtest.engines.india import IndiaBacktestEngine
from config.india_settings import (
    INDIA_DEFAULT_SYMBOL,
    INDIA_DEFAULT_INTERVAL,
    INDIA_SIGNAL_THRESHOLD,
    INDIA_ATR_SL_MULT,
    INDIA_ATR_TP_MULT,
    INITIAL_CAPITAL_INR,
)
from rich.console import Console

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="India NSE backtest runner")
    parser.add_argument("--symbol",    default=INDIA_DEFAULT_SYMBOL,   help="NSE symbol e.g. NIFTY50 BANKNIFTY")
    parser.add_argument("--interval",  default=INDIA_DEFAULT_INTERVAL,  help="15m 1h 1d")
    parser.add_argument("--bars",      default=1400, type=int,          help="Historical candles (15m: max ~1400, 1h: ~730)")
    parser.add_argument("--threshold", default=INDIA_SIGNAL_THRESHOLD, type=float, help="Min signal probability")
    parser.add_argument("--sl-mult",   default=INDIA_ATR_SL_MULT,      type=float, help="ATR SL multiplier")
    parser.add_argument("--tp-mult",   default=INDIA_ATR_TP_MULT,      type=float, help="R-multiple for TP")
    parser.add_argument("--capital",   default=INITIAL_CAPITAL_INR,    type=float, help="Starting capital INR")
    parser.add_argument("--timeout",   default=0,   type=int,           help="Max candles to hold (0=no timeout)")
    parser.add_argument("--vix",       default=15.0, type=float,        help="India VIX to use for backtest")
    args = parser.parse_args()

    console.print(f"\n[bold cyan]India Backtest — {args.symbol} {args.interval}[/]")
    console.print(f"[dim]Threshold={args.threshold}%  SL={args.sl_mult}×ATR  TP={args.tp_mult}R  Capital=₹{args.capital:,.0f}[/]\n")

    # Fetch live VIX if not overridden
    vix = args.vix
    try:
        client = NSEClient()
        vix = client.get_india_vix()
        console.print(f"[dim]Live India VIX: {vix:.2f}[/]\n")
    except Exception:
        console.print(f"[yellow]VIX fetch failed — using {vix:.1f}[/]\n")

    bt = IndiaBacktestEngine(
        symbol      = args.symbol,
        interval    = args.interval,
        bars        = args.bars,
        threshold   = args.threshold,
        atr_sl_mult = args.sl_mult,
        atr_tp_mult = args.tp_mult,
        capital     = args.capital,
        vix         = vix,
        max_timeout = args.timeout,
    )

    result = bt.run()
    bt.print_report(result)


if __name__ == "__main__":
    main()
