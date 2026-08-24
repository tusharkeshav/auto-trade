# ─────────────────────────────────────────────────────────────────
#  run_expiry_pin_backtest.py  —  Options expiry pinning backtest
#
#  Usage:
#      python run_expiry_pin_backtest.py
#      python run_expiry_pin_backtest.py --symbol BANKNIFTY --bars 500
#      python run_expiry_pin_backtest.py --symbol NIFTY50 --sl-mult 1.5 --tp-mult 2.0
#
#  Strategy: trade toward max pain (5-day EMA proxy) on Wed/Thu
#  Data:     daily OHLCV from yfinance (free, no limit)
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.engines.expiry_pin import ExpiryPinEngine
from config.india_settings import (
    INDIA_ATR_SL_MULT,
    INDIA_ATR_TP_MULT,
    INITIAL_CAPITAL_INR,
)
from rich.console import Console

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Options expiry pin backtest runner")
    parser.add_argument("--symbol",      default="NIFTY50",           help="NSE symbol: NIFTY50 or BANKNIFTY")
    parser.add_argument("--bars",        default=500,    type=int,     help="Daily bars (500 ≈ 2 years)")
    parser.add_argument("--capital",     default=INITIAL_CAPITAL_INR, type=float, help="Starting capital INR")
    parser.add_argument("--sl-mult",     default=INDIA_ATR_SL_MULT,   type=float, help="ATR SL multiplier")
    parser.add_argument("--tp-mult",     default=INDIA_ATR_TP_MULT,   type=float, help="R-multiple for TP")
    parser.add_argument("--risk-pct",    default=1.0,   type=float,   help="Risk per trade % of capital")
    parser.add_argument("--min-dist",    default=0.3,   type=float,   help="Min %% from max pain to trade")
    args = parser.parse_args()

    console.print(f"\n[bold cyan]Expiry Pin Backtest — {args.symbol} (Daily)[/]")
    console.print(
        f"[dim]SL={args.sl_mult}×ATR  TP={args.tp_mult}R  "
        f"Risk={args.risk_pct}%/trade  MinDist={args.min_dist}%  "
        f"Capital=₹{args.capital:,.0f}[/]\n"
    )
    console.print("[dim]Synthetic max pain = 5-day EMA of close (proxy). "
                  "Real forward test uses live NSE options chain.[/]\n")

    engine = ExpiryPinEngine(
        symbol       = args.symbol,
        bars         = args.bars,
        capital      = args.capital,
        atr_sl_mult  = args.sl_mult,
        atr_tp_mult  = args.tp_mult,
        risk_pct     = args.risk_pct,
        min_dist_pct = args.min_dist,
    )

    result = engine.run()
    engine.print_report(result)


if __name__ == "__main__":
    main()
