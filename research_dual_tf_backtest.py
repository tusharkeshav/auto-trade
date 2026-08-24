"""
research_dual_tf_backtest.py
─────────────────────────────────────────────────────────────────────────────
Phase 1D — Option C Research: Combined 15m + 30m Portfolio Simulation

Simulates the COMBINED portfolio running both timeframes simultaneously:
  - Each 15m candle close: score 15m signal → open if tradeable + no open pos
  - Each 30m candle close: score 30m signal → open ONLY if 15m pos NOT open
  - One position per symbol at a time (position guard active)
  - Same macro rules, same threshold, same All-In/All-Out exit

Compares:
  │  15m alone
  │  30m alone
  │  Combined (15m priority + 30m fills gaps)

Usage:
    source .venv/bin/activate && python research_dual_tf_backtest.py
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
import json
import math
import warnings
from dataclasses import replace, dataclass
from pathlib import Path
from typing import Optional

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

THRESHOLD    = 48.0
ATR_SL_MULT  = 1.25
ATR_TP_MULT  = 1.5
WARMUP_15M   = 200     # warmup candles on 15m
WARMUP_30M   = 200     # warmup candles on 30m (same indicator periods)


# ─── Resample ────────────────────────────────────────────────────────────────

def resample_to_30m(df_15m: pd.DataFrame) -> pd.DataFrame:
    df = df_15m.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    return df.resample("30min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()


# ─── Trade event (used in the combined simulation) ─────────────────────────

@dataclass
class SimTrade:
    symbol:      str
    timeframe:   str    # "15m" or "30m"
    direction:   str
    entry_time:  pd.Timestamp
    exit_time:   pd.Timestamp
    entry_price: float
    exit_price:  float
    pnl:         float
    exit_type:   str


# ─── Single-timeframe simulation (helper, reusable) ────────────────────────

def simulate_tf(df: pd.DataFrame, symbol: str, tf_label: str,
                warmup: int, existing_blocked: set[pd.Timestamp]) -> list[SimTrade]:
    """
    Simulate trades on df, skipping entries where timestamp is in existing_blocked
    (used to block 30m entries when 15m position is already open).
    Returns list of SimTrade.
    """
    scorer = SignalScorer(symbol=symbol, long_threshold=THRESHOLD)
    trades: list[SimTrade] = []
    in_position = False
    i = warmup

    while i < len(df) - 1:
        row = df.iloc[i]

        if not in_position:
            # Position guard: skip if this candle's time was blocked by the other TF
            if df.index[i] in existing_blocked:
                i += 1
                continue

            signal = scorer.score(row)
            if signal.direction in ("LONG", "SHORT") and signal.probability >= THRESHOLD:
                atr     = row["atr"] if not math.isnan(row["atr"]) else signal.entry_price * 0.004
                sl_dist = atr * ATR_SL_MULT
                entry   = signal.entry_price
                if sl_dist == 0:
                    i += 1
                    continue

                if signal.direction == "LONG":
                    sl = round(entry - sl_dist, 2)
                    tp = round(entry + sl_dist * ATR_TP_MULT, 2)
                else:
                    sl = round(entry + sl_dist, 2)
                    tp = round(entry - sl_dist * ATR_TP_MULT, 2)

                # Walk forward
                for j in range(i + 1, len(df)):
                    candle = df.iloc[j]
                    hi, lo = candle["high"], candle["low"]

                    if signal.direction == "LONG":
                        if lo <= sl:
                            pnl = (sl - entry) * (row["close"] / sl)  # approx
                            size = (INITIAL_CAPITAL_USDT * MAX_RISK_PER_TRADE_PCT / 100) / sl_dist
                            pnl  = (sl - entry) * size
                            trades.append(SimTrade(
                                symbol, tf_label, signal.direction,
                                df.index[i], df.index[j],
                                entry, sl, pnl, "STOP_LOSS"
                            ))
                            i += (j - i)
                            break
                        if hi >= tp:
                            size = (INITIAL_CAPITAL_USDT * MAX_RISK_PER_TRADE_PCT / 100) / sl_dist
                            pnl  = (tp - entry) * size
                            trades.append(SimTrade(
                                symbol, tf_label, signal.direction,
                                df.index[i], df.index[j],
                                entry, tp, pnl, "TAKE_PROFIT"
                            ))
                            i += (j - i)
                            break
                    else:
                        if hi >= sl:
                            size = (INITIAL_CAPITAL_USDT * MAX_RISK_PER_TRADE_PCT / 100) / sl_dist
                            pnl  = (entry - sl) * size
                            pnl  = -pnl  # loss
                            trades.append(SimTrade(
                                symbol, tf_label, signal.direction,
                                df.index[i], df.index[j],
                                entry, sl, pnl, "STOP_LOSS"
                            ))
                            i += (j - i)
                            break
                        if lo <= tp:
                            size = (INITIAL_CAPITAL_USDT * MAX_RISK_PER_TRADE_PCT / 100) / sl_dist
                            pnl  = (entry - tp) * size
                            trades.append(SimTrade(
                                symbol, tf_label, signal.direction,
                                df.index[i], df.index[j],
                                entry, tp, pnl, "TAKE_PROFIT"
                            ))
                            i += (j - i)
                            break

        i += 1

    return trades


# ─── Metrics helper ──────────────────────────────────────────────────────────

def compute_metrics(trades: list[SimTrade], years: float, label: str) -> dict:
    if not trades:
        return {"label": label, "trades": 0, "trades_per_yr": 0,
                "win_rate": 0, "pf": 0, "max_dd_pct": 0, "total_pnl_pct": 0}

    wins   = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gw     = sum(t.pnl for t in wins)
    gl     = abs(sum(t.pnl for t in losses))
    pf     = round(gw / gl, 2) if gl > 0 else 999.0

    # Equity curve max DD
    equity = INITIAL_CAPITAL_USDT
    peak   = INITIAL_CAPITAL_USDT
    max_dd = 0.0
    for t in sorted(trades, key=lambda x: x.entry_time):
        equity += t.pnl
        peak    = max(peak, equity)
        dd      = (peak - equity) / peak * 100
        max_dd  = max(max_dd, dd)

    total_pnl_pct = (sum(t.pnl for t in trades) / INITIAL_CAPITAL_USDT) * 100

    return {
        "label":          label,
        "trades":         len(trades),
        "trades_per_yr":  round(len(trades) / years, 1),
        "win_rate":       round(len(wins) / len(trades) * 100, 1),
        "pf":             pf,
        "max_dd_pct":     round(max_dd, 1),
        "total_pnl_pct":  round(total_pnl_pct, 1),
    }


# ─── Combined simulation for one asset ───────────────────────────────────────

def run_combined_simulation(symbol: str, csv_path: str) -> dict:
    console.print(f"\n[bold cyan]▶  {symbol}  —  Combined Dual-TF Simulation[/]")

    # Load data
    df_15m_raw = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df_15m_raw.index = pd.to_datetime(df_15m_raw.index, utc=True)
    df_30m_raw = resample_to_30m(df_15m_raw)

    console.print("  [dim]Computing 15m indicators...[/]")
    df_15m = add_all_indicators(df_15m_raw.copy())
    console.print("  [dim]Computing 30m indicators...[/]")
    df_30m = add_all_indicators(df_30m_raw.copy())

    # Align start dates
    start = max(df_15m.index[WARMUP_15M], df_30m.index[WARMUP_30M])
    df_15m = df_15m[df_15m.index >= start]
    df_30m = df_30m[df_30m.index >= start]

    days  = (df_15m.index[-1] - df_15m.index[0]).days
    years = days / 365.25

    # ── Step 1: Run 15m alone ─────────────────────────────────────────────────
    console.print("  [dim]Simulating 15m alone...[/]")
    trades_15m = simulate_tf(df_15m, symbol, "15m", 0, set())

    # ── Step 2: Run 30m alone ─────────────────────────────────────────────────
    console.print("  [dim]Simulating 30m alone...[/]")
    trades_30m_alone = simulate_tf(df_30m, symbol, "30m", 0, set())

    # ── Step 3: Combined — 15m trades block 30m entries ──────────────────────
    # Build a set of timestamp ranges blocked by active 15m positions
    console.print("  [dim]Simulating 30m with position guard (15m priority)...[/]")
    blocked_30m_times: set[pd.Timestamp] = set()
    for t in trades_15m:
        # Block every 30m timestamp between entry and exit of each 15m trade
        t_start = t.entry_time.floor("30min")
        t_end   = t.exit_time.ceil("30min")
        ts = t_start
        while ts <= t_end:
            blocked_30m_times.add(ts)
            ts = ts + pd.Timedelta(minutes=30)

    trades_30m_guarded = simulate_tf(df_30m, symbol, "30m_guarded", 0, blocked_30m_times)

    # Combined = 15m trades + 30m trades that weren't blocked
    trades_combined = trades_15m + trades_30m_guarded
    trades_combined.sort(key=lambda t: t.entry_time)

    m15 = compute_metrics(trades_15m, years, f"{symbol} 15m alone")
    m30 = compute_metrics(trades_30m_alone, years, f"{symbol} 30m alone")
    mc  = compute_metrics(trades_combined, years, f"{symbol} Combined")

    return {
        "symbol":        symbol,
        "years":         round(years, 1),
        "metrics_15m":   m15,
        "metrics_30m":   m30,
        "metrics_combined": mc,
        "additive_30m_trades":     len(trades_30m_guarded),
        "additive_30m_per_yr":     round(len(trades_30m_guarded) / years, 1),
        "blocked_30m_trades":      len(trades_30m_alone) - len(trades_30m_guarded),
    }


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    console.print("\n[bold magenta]═══  Option C Research — Phase 1D: Combined Dual-TF Simulation  ═══[/]")
    console.print(f"[dim]Strategy: 15m fires first (priority), 30m fills gaps only[/]")
    console.print(f"[dim]One position per symbol at a time — position guard active[/]\n")

    ASSETS = {
        "BTCUSDT": "data/historical/BTCUSDT_15m_180000.csv",
        "ETHUSDT": "data/historical/ETHUSDT_15m_87000.csv",
    }

    all_results = []
    all_15m_trades = []
    all_30m_guarded_trades = []

    for symbol, csv_path in ASSETS.items():
        if not Path(csv_path).exists():
            console.print(f"[red]  ✗ {csv_path} not found — skipping[/]")
            continue
        r = run_combined_simulation(symbol, csv_path)
        all_results.append(r)

    # ─── Print comparison table ───────────────────────────────────────────────
    table = Table(title="\nDual-TF Backtest Comparison", border_style="cyan")
    table.add_column("Configuration",  style="bold white", min_width=22)
    table.add_column("Trades",         justify="right")
    table.add_column("Trades/Yr",      justify="right")
    table.add_column("Win Rate",       justify="right")
    table.add_column("PF",             justify="right")
    table.add_column("Max DD",         justify="right")
    table.add_column("Total PnL",      justify="right")

    def add_row(m: dict):
        pf_color = "green" if m["pf"] >= 1.30 else ("yellow" if m["pf"] >= 1.0 else "red")
        dd_color = "green" if m["max_dd_pct"] < 15 else "red"
        table.add_row(
            m["label"],
            str(m["trades"]),
            str(m["trades_per_yr"]),
            f"{m['win_rate']:.1f}%",
            f"[{pf_color}]{m['pf']:.2f}[/]",
            f"[{dd_color}]{m['max_dd_pct']:.1f}%[/]",
            f"{m['total_pnl_pct']:+.1f}%",
        )

    for r in all_results:
        add_row(r["metrics_15m"])
        add_row(r["metrics_30m"])
        add_row(r["metrics_combined"])
        table.add_section()

    console.print(table)

    # ─── Go/No-Go assessment ──────────────────────────────────────────────────
    console.print("\n[bold]Go/No-Go criteria for combined (Phase 2 threshold):[/]")
    console.print("  Combined PF ≥ 1.30  |  Combined Max DD < 15%  |  Combined Trades/yr ≥ 30")

    for r in all_results:
        mc = r["metrics_combined"]
        go = mc["pf"] >= 1.30 and mc["max_dd_pct"] < 15 and mc["trades_per_yr"] >= 30
        icon = "✅" if go else ("⚠️ " if mc["pf"] >= 1.0 else "❌")
        console.print(
            f"  {icon}  {r['symbol']} Combined: "
            f"PF={mc['pf']}  WR={mc['win_rate']}%  DD={mc['max_dd_pct']}%  "
            f"Trades/yr={mc['trades_per_yr']}"
        )
        console.print(
            f"      ↳ 15m contributed {r['metrics_15m']['trades']} trades, "
            f"30m added {r['additive_30m_trades']} ({r['blocked_30m_trades']} blocked by position guard)"
        )

    # Save results
    out = Path("research_out")
    out.mkdir(exist_ok=True)
    (out / "dual_tf_backtest.json").write_text(json.dumps(
        [{**r, "metrics_15m": r["metrics_15m"], "metrics_30m": r["metrics_30m"],
          "metrics_combined": r["metrics_combined"]} for r in all_results],
        indent=2, default=str
    ))
    console.print(f"\n[dim]Results saved → research_out/dual_tf_backtest.json[/]")


if __name__ == "__main__":
    main()
