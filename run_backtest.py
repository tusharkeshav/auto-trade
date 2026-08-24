# ─────────────────────────────────────────────────────────────────
#  run_backtest.py
#  Runs backtest on multiple symbols and prints reports.
#
#  Usage:
#      source .venv/bin/activate && python run_backtest.py
# ─────────────────────────────────────────────────────────────────

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from backtest import BacktestEngine

console = Console()

# ── Backtest config ───────────────────────────────────────────────
SYMBOLS   = ["BTCUSDT"]
INTERVAL  = "15m"    # 15m candles
CANDLES   = 87000     # ~2.5 years
THRESHOLD = 46.0     # min probability to enter
ATR_MULT  = 1.0      # Stop loss distance multiplier


def main():
    console.print(
        f"\n[bold cyan]🔁 BACKTEST — 30 Days  |  {INTERVAL} candles  |  "
        f"Threshold ≥{THRESHOLD}%[/]\n"
        f"[dim]Symbols: {', '.join(SYMBOLS)}[/]\n"
    )
    for symbol in SYMBOLS:
        console.print(f"[yellow]⏳  Running {symbol}...[/]")
        bt     = BacktestEngine(symbol=symbol, interval=INTERVAL, candles=CANDLES, threshold=THRESHOLD, atr_sl_mult=ATR_MULT)
        result = bt.run()
        bt.print_report(result)


if __name__ == "__main__":
    main()

