"""
research_30m_standalone.py
─────────────────────────────────────────────────────────────────────────────
Phase 1B — Option C Research: Does 30m have a standalone edge?

Methodology:
  - Resample existing 15m OHLCV → 30m using pandas resample()
  - Re-compute all indicators on the 30m frame
  - Run BacktestEngine with the SAME locked macro rules (ADX ≤ 25, 16-24 UTC)
  - Report: Trade count, Win Rate, Profit Factor, Max Drawdown, Trades/Year

Usage:
    source .venv/bin/activate && python research_30m_standalone.py
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
import math
import warnings
from dataclasses import replace
from pathlib import Path

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from indicators                   import add_all_indicators
from probability.signal_scorer    import SignalScorer
from config.settings              import (
    INITIAL_CAPITAL_USDT, MAX_RISK_PER_TRADE_PCT,
    MACRO_MAX_ADX, MACRO_SESSION_START, MACRO_SESSION_END,
)

console = Console()

# ─── Config ──────────────────────────────────────────────────────────────────

ASSETS = {
    "BTCUSDT": "data/historical/BTCUSDT_15m_180000.csv",   # 5 years
    "ETHUSDT": "data/historical/ETHUSDT_15m_87000.csv",    # ~2.5 years
}

THRESHOLD    = 48.0    # locked signal threshold
ATR_SL_MULT  = 1.25   # locked SL multiplier
ATR_TP_MULT  = 1.5    # 1.5R take profit (All-In, All-Out)
WARMUP       = 200     # candles for indicator warm-up

# ─── Resample 15m → 30m ──────────────────────────────────────────────────────

def resample_to_30m(df_15m: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 15m OHLCV candles into 30m candles.
    Two consecutive 15m candles → one 30m candle.
    """
    df = df_15m.copy()
    df.index = pd.to_datetime(df.index, utc=True)

    df_30m = df.resample("30min").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()

    logger.info(f"Resampled {len(df_15m):,} 15m → {len(df_30m):,} 30m candles  "
                f"({df_30m.index[0].date()} → {df_30m.index[-1].date()})")
    return df_30m


# ─── Backtest core (mirrors BacktestEngine but standalone) ───────────────────

def run_backtest_30m(symbol: str, csv_path: str) -> dict:
    """
    Load 15m data, resample to 30m, compute indicators, run simulation.
    Returns a summary dict.
    """
    console.print(f"\n[bold cyan]▶  {symbol}  —  30m standalone backtest[/]")

    # 1. Load raw 15m
    df_15m = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(df_15m):,} 15m rows from {csv_path}")

    # 2. Resample → 30m
    df = resample_to_30m(df_15m)

    # 3. Indicators on 30m frame
    console.print("  [dim]Computing indicators on 30m frame...[/]")
    df = add_all_indicators(df)

    # 4. Simulate
    scorer  = SignalScorer(symbol=symbol, long_threshold=THRESHOLD)
    capital = INITIAL_CAPITAL_USDT
    cash    = capital

    trades       = []
    in_position  = False
    i            = WARMUP
    total_candles = len(df)

    period_start = df.index[WARMUP]
    period_end   = df.index[-1]
    days = (period_end - period_start).days
    years = days / 365.25

    while i < total_candles - 1:
        row = df.iloc[i]

        if not in_position:
            signal = scorer.score(row)

            # Only open if signal passes the SAME macro gates (already in scorer)
            if signal.direction in ("LONG", "SHORT") and signal.probability >= THRESHOLD:
                atr     = row["atr"] if not math.isnan(row["atr"]) else signal.entry_price * 0.004
                sl_dist = atr * ATR_SL_MULT
                entry   = signal.entry_price

                if signal.direction == "LONG":
                    sl = round(entry - sl_dist, 2)
                    tp = round(entry + sl_dist * ATR_TP_MULT, 2)
                else:
                    sl = round(entry + sl_dist, 2)
                    tp = round(entry - sl_dist * ATR_TP_MULT, 2)

                signal = replace(signal, stop_loss=sl, take_profit1=tp, take_profit2=tp, risk_amount=round(sl_dist, 2))

                risk_budget   = cash * (MAX_RISK_PER_TRADE_PCT / 100)
                risk_per_unit = sl_dist
                if risk_per_unit == 0:
                    i += 1
                    continue
                size = risk_budget / risk_per_unit

                # Walk forward to find SL/TP hit
                hit = None
                for j in range(i + 1, total_candles):
                    candle = df.iloc[j]
                    hi, lo = candle["high"], candle["low"]
                    candles_held = j - i

                    if signal.direction == "LONG":
                        if lo <= sl:
                            hit = ("STOP_LOSS",  sl, candles_held, j)
                            break
                        if hi >= tp:
                            hit = ("TAKE_PROFIT", tp, candles_held, j)
                            break
                    else:
                        if hi >= sl:
                            hit = ("STOP_LOSS",  sl, candles_held, j)
                            break
                        if lo <= tp:
                            hit = ("TAKE_PROFIT", tp, candles_held, j)
                            break

                if hit:
                    exit_type, exit_price, candles_held, j = hit
                    if signal.direction == "LONG":
                        pnl = (exit_price - entry) * size
                    else:
                        pnl = (entry - exit_price) * size

                    cash += pnl
                    trades.append({
                        "entry_time":   df.index[i],
                        "exit_time":    df.index[j],
                        "direction":    signal.direction,
                        "exit_type":    exit_type,
                        "entry_price":  entry,
                        "exit_price":   exit_price,
                        "pnl":          pnl,
                        "probability":  signal.probability,
                        "candles_held": candles_held,
                    })
                    i += candles_held
                    continue

        i += 1

    # ─── Compute metrics ─────────────────────────────────────────────────────
    if not trades:
        return {
            "symbol": symbol, "timeframe": "30m",
            "trades": 0, "trades_per_year": 0,
            "win_rate": 0, "profit_factor": 0,
            "max_dd_pct": 0, "total_pnl_pct": 0,
            "days": days, "years": round(years, 1),
        }

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    gross_win  = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 999.0

    # Max drawdown from equity curve
    equity = capital
    peak   = capital
    max_dd = 0.0
    for t in trades:
        equity += t["pnl"]
        peak    = max(peak, equity)
        dd      = (peak - equity) / peak * 100
        max_dd  = max(max_dd, dd)

    return {
        "symbol":         symbol,
        "timeframe":      "30m",
        "trades":         len(trades),
        "trades_per_year": round(len(trades) / years, 1),
        "win_rate":       round(len(wins) / len(trades) * 100, 1),
        "profit_factor":  pf,
        "max_dd_pct":     round(max_dd, 1),
        "total_pnl_pct":  round((cash - capital) / capital * 100, 1),
        "days":           days,
        "years":          round(years, 1),
    }


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    console.print("\n[bold magenta]═══  Option C Research — Phase 1B: 30m Standalone Backtest  ═══[/]")
    console.print(f"[dim]Macro Shield: ADX ≤ {MACRO_MAX_ADX}  |  Session: {MACRO_SESSION_START}:00–{MACRO_SESSION_END}:00 UTC[/]")
    console.print(f"[dim]Threshold: {THRESHOLD}  |  SL: {ATR_SL_MULT}×ATR  |  TP: {ATR_TP_MULT}×R  (All-In/All-Out)[/]\n")

    results = []
    for symbol, csv_path in ASSETS.items():
        if not Path(csv_path).exists():
            console.print(f"[red]  ✗ {csv_path} not found — skipping {symbol}[/]")
            continue
        r = run_backtest_30m(symbol, csv_path)
        results.append(r)

    # ─── Print results table ──────────────────────────────────────────────────
    table = Table(title="\n30m Standalone Results", border_style="cyan")
    table.add_column("Asset",        style="bold white")
    table.add_column("Period",       style="dim")
    table.add_column("Trades",       justify="right")
    table.add_column("Trades/Yr",    justify="right")
    table.add_column("Win Rate",     justify="right")
    table.add_column("PF",           justify="right")
    table.add_column("Max DD",       justify="right")
    table.add_column("Total PnL",    justify="right")

    for r in results:
        pf_color  = "green" if r["profit_factor"] >= 1.3 else "red"
        dd_color  = "green" if r["max_dd_pct"] < 15 else "red"
        table.add_row(
            r["symbol"],
            f"{r['years']}y",
            str(r["trades"]),
            str(r["trades_per_year"]),
            f"{r['win_rate']:.1f}%",
            f"[{pf_color}]{r['profit_factor']:.2f}[/]",
            f"[{dd_color}]{r['max_dd_pct']:.1f}%[/]",
            f"{r['total_pnl_pct']:+.1f}%",
        )

    console.print(table)

    console.print("\n[bold]Go/No-Go criteria for 30m standalone:[/]")
    console.print("  PF ≥ 1.20  |  Win Rate ≥ 45%  |  Max DD < 15%")

    for r in results:
        go = r["profit_factor"] >= 1.20 and r["win_rate"] >= 45 and r["max_dd_pct"] < 15
        icon = "✅" if go else "❌"
        console.print(f"  {icon}  {r['symbol']}: PF={r['profit_factor']}  WR={r['win_rate']}%  DD={r['max_dd_pct']}%")

    # Save for summary script
    import json
    out = Path("research_out")
    out.mkdir(exist_ok=True)
    (out / "standalone_30m.json").write_text(json.dumps(results, indent=2))
    console.print(f"\n[dim]Results saved → research_out/standalone_30m.json[/]")


if __name__ == "__main__":
    main()
