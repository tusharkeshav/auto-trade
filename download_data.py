# ─────────────────────────────────────────────────────────────────
#  download_data.py
#
#  One-shot historical data dump.
#  Run this once (or daily) to refresh local cache.
#
#  Usage:
#      source .venv/bin/activate && python download_data.py
#
#  What it does:
#    - Fetches max 1000 candles per symbol × interval from Binance
#    - Saves to data/historical/ as CSV (or Parquet if pyarrow installed)
#    - Skips files that are < 12h old (use --force to override)
# ─────────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from rich.console import Console
from rich.table   import Table
from rich         import box

from data.binance_client import BinanceClient
from data.cache          import DataCache
from indicators          import add_all_indicators

SYMBOLS    = ["BTCUSDT"]
INTERVALS  = ["15m"]
CANDLES    = 180000       # ~5 years of 15m data
REFRESH_H  = 12          # re-download if cache is older than this

console = Console()


def main(force: bool = False):
    client = BinanceClient()
    cache  = DataCache()

    console.print("\n[bold cyan]📥  Data Downloader — Fetching historical OHLCV[/]\n")

    jobs = [(sym, iv) for sym in SYMBOLS for iv in INTERVALS]
    done, skipped, failed = [], [], []

    for symbol, interval in jobs:
        age_h = cache.age_hours(symbol, interval)
        if not force and age_h < REFRESH_H:
            console.print(
                f"  [dim]⏭  {symbol} {interval:>3}  — cached ({age_h:.1f}h old, < {REFRESH_H}h)[/]"
            )
            skipped.append((symbol, interval))
            continue

        console.print(f"  [yellow]⬇  {symbol} {interval:>3}  — fetching {CANDLES} candles...[/]", end=" ")
        try:
            df = client.get_ohlcv(symbol, interval, CANDLES)
            df = add_all_indicators(df)   # pre-compute so backtest is instant
            path = cache.save(df, symbol, interval)
            rows = len(df)
            start = df.index[0].strftime("%Y-%m-%d")
            end   = df.index[-1].strftime("%Y-%m-%d")
            console.print(f"[green]✅  {rows} rows  ({start} → {end})[/]")
            done.append((symbol, interval, rows, path))
        except Exception as e:
            console.print(f"[red]❌  FAILED: {e}[/]")
            failed.append((symbol, interval))

    # ── Summary table ─────────────────────────────────────────────
    console.print()
    t = Table(
        "File", "Symbol", "Interval", "Rows", "Size", "Age",
        box=box.SIMPLE_HEAD, header_style="bold cyan",
    )
    for item in cache.list_cached():
        age_str = f"{item['age_h']}h"
        t.add_row(
            item["file"],
            item["symbol"],
            item["interval"],
            "—",
            f"{item['size_kb']} KB",
            f"[green]{age_str}[/]" if item["age_h"] < REFRESH_H else f"[yellow]{age_str}[/]",
        )
    console.print(t)

    console.print(
        f"[green]Downloaded: {len(done)}[/]  "
        f"[dim]Skipped: {len(skipped)}[/]  "
        f"[red]Failed: {len(failed)}[/]\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Re-download even if cache is fresh")
    args = p.parse_args()
    main(force=args.force)
