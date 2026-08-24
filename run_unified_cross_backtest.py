# ─────────────────────────────────────────────────────────────────
#  run_unified_cross_backtest.py  —  Unified Cross Strategy Backtest Runner
#
#  Usage:
#      python run_unified_cross_backtest.py
#      python run_unified_cross_backtest.py --symbol BANKNIFTY --interval 15m --bars 1400
#      python run_unified_cross_backtest.py --symbol NIFTY50 --vix 18.5 --capital 500000
#
#  Strategy: "Momentum Crosses Mean" (VWAP Pullback + BB Reversal Cross)
#            with 3 Hardened Shields & 1:2.5 Asymmetric R-multiple.
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.engines.unified_cross import UnifiedCrossBacktestEngine
from config.india_settings import (
    INDIA_DEFAULT_SYMBOL,
    INDIA_DEFAULT_INTERVAL,
    INDIA_DEFAULT_BARS,
    INITIAL_CAPITAL_INR,
    INDIA_SIGNAL_THRESHOLD,
)
from rich.console import Console

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Cross Strategy Backtest Runner")
    parser.add_argument("--symbol",    default=INDIA_DEFAULT_SYMBOL,   help="NSE symbol: NIFTY50 or BANKNIFTY")
    parser.add_argument("--interval",  default=INDIA_DEFAULT_INTERVAL, help="Candle timeframe: 15m, 1h, 1d")
    parser.add_argument("--bars",      default=1400,                   type=int,   help="Number of historical candles (max ~2000 for 15m in yfinance)")
    parser.add_argument("--capital",   default=INITIAL_CAPITAL_INR,    type=float, help="Starting capital INR (default: ₹5,00,000)")
    parser.add_argument("--threshold", default=INDIA_SIGNAL_THRESHOLD, type=float, help="Min probability for entry (default: 50)")
    parser.add_argument("--sl-mult",   default=1.0,                    type=float, help="ATR stop-loss multiplier (default: 1.0)")
    parser.add_argument("--tp-mult",   default=2.5,                    type=float, help="ATR take-profit R-multiple (default: 2.5)")
    parser.add_argument("--vix",       default=15.0,                   type=float, help="India VIX level (default: 15.0)")
    parser.add_argument("--timeout",   default=0,                      type=int,   help="Max candles to hold before time exit (0=none)")
    args = parser.parse_args()

    console.print(f"\n[bold cyan]Unified Cross Backtest — {args.symbol} ({args.interval})[/]")
    console.print(
        f"[dim]SL={args.sl_mult}×ATR  TP={args.tp_mult}R  "
        f"Threshold={args.threshold}%  VIX={args.vix:.1f}  "
        f"Capital=₹{args.capital:,.0f}[/]\n"
    )
    console.print("[dim]Strategy: VWAP/EMA Pullback Cross + BB Reversal Cross (1:2.5 Asymmetric R)[/]\n")

    engine = UnifiedCrossBacktestEngine(
        symbol      = args.symbol,
        interval    = args.interval,
        bars        = args.bars,
        threshold   = args.threshold,
        atr_sl_mult = args.sl_mult,
        atr_tp_mult = args.tp_mult,
        capital     = args.capital,
        vix         = args.vix,
        max_timeout = args.timeout,
    )

    result = engine.run()
    engine.print_report(result)


if __name__ == "__main__":
    main()
