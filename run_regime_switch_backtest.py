# ─────────────────────────────────────────────────────────────────
#  run_regime_switch_backtest.py  —  VIX Regime-Switching Backtest Runner
#
#  Usage:
#      python run_regime_switch_backtest.py
#      python run_regime_switch_backtest.py --symbol BANKNIFTY --interval 15m --bars 1400
#      python run_regime_switch_backtest.py --symbol NIFTY50 --vix 14.5 --capital 500000
#
#  Strategy: Dynamically switch/blend between Momentum and Mean-Reversion
#            based on India VIX level (Low/Transition/High Vol zones).
#  Data:     15m / 1h / 1d OHLCV from yfinance (free)
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.engines.regime_switch import RegimeSwitchBacktestEngine
from config.india_settings import (
    INDIA_DEFAULT_SYMBOL,
    INDIA_DEFAULT_INTERVAL,
    INDIA_DEFAULT_BARS,
    INITIAL_CAPITAL_INR,
    INDIA_SIGNAL_THRESHOLD,
    INDIA_ATR_SL_MULT,
    INDIA_ATR_TP_MULT,
)
from rich.console import Console

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="VIX Regime-Switching Strategy Backtest Runner")
    parser.add_argument("--symbol",    default=INDIA_DEFAULT_SYMBOL,   help="NSE symbol: NIFTY50 or BANKNIFTY")
    parser.add_argument("--interval",  default=INDIA_DEFAULT_INTERVAL, help="Candle timeframe: 15m, 1h, 1d")
    parser.add_argument("--bars",      default=1400,                   type=int,   help="Number of historical candles (max ~2000 for 15m in yfinance)")
    parser.add_argument("--capital",   default=INITIAL_CAPITAL_INR,    type=float, help="Starting capital INR (default: ₹5,00,000)")
    parser.add_argument("--threshold", default=INDIA_SIGNAL_THRESHOLD, type=float, help="Min probability for entry (default: 50)")
    parser.add_argument("--sl-mult",   default=INDIA_ATR_SL_MULT,      type=float, help="ATR stop-loss multiplier (default: 1.25)")
    parser.add_argument("--tp-mult",   default=INDIA_ATR_TP_MULT,      type=float, help="ATR take-profit R-multiple (default: 1.5)")
    parser.add_argument("--vix",       default=15.0,                   type=float, help="India VIX level (default: 15.0)")
    parser.add_argument("--timeout",   default=0,                      type=int,   help="Max candles to hold before time exit (0=none)")
    args = parser.parse_args()

    console.print(f"\n[bold cyan]VIX Regime-Switching Backtest — {args.symbol} ({args.interval})[/]")
    console.print(
        f"[dim]SL={args.sl_mult}×ATR  TP={args.tp_mult}R  "
        f"Threshold={args.threshold}%  VIX={args.vix:.1f}  "
        f"Capital=₹{args.capital:,.0f}[/]\n"
    )
    console.print("[dim]Regimes: VIX < 18 (Momentum) | VIX 18-25 (Hybrid Blend) | VIX > 25 (Mean-Reversion)[/]\n")

    engine = RegimeSwitchBacktestEngine(
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
