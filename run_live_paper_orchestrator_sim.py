# ─────────────────────────────────────────────────────────────────
#  run_live_paper_orchestrator_sim.py
#  Standalone Forward Test Replay Simulation for Live Paper Orchestrator.
#
#  Features:
#    • Day-by-Day Historical Bar Replay (1-Year Forward & 2-Year Master Replay)
#    • Exact Indian CNC Delivery Tax Simulation (STT, GST, Stamp, SEBI, DP)
#    • Multi-Parameter Grid Comparison for GOLDBEES & Pullback Engine
#    • Zero Impact on Active Live Trading Database
# ─────────────────────────────────────────────────────────────────

import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.india_costs import calculate_round_trip_cost
from indicators import add_all_indicators
from config.india_settings import INITIAL_CAPITAL_INR

console = Console()

BENCHMARK_SYMBOL = "^NSEI"
SAFE_ASSET_SYMBOL = "GOLDBEES.NS"
ETFS_ALL = ["NIFTYBEES.NS", "BANKBEES.NS", "CPSEETF.NS", "ITBEES.NS", "AUTOBEES.NS", "PHARMABEES.NS"]
STOCKS_ALL = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "LT.NS",
    "BHARTIARTL.NS", "SBIN.NS", "SUNPHARMA.NS", "NTPC.NS"
]


@dataclass
class SimTrade:
    symbol: str
    strategy: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float
    taxes: float
    net_pnl: float
    net_pnl_pct: float
    exit_reason: str
    bars_held: int


@dataclass
class SimConfig:
    name: str
    gold_sl_pct: float         # e.g. 0.08 = 8%
    gold_tp_pct: float         # e.g. 0.15 = 15%
    pullback_tp_mult: float    # e.g. 4.0
    be_lock_r: float           # e.g. 2.0
    max_hold_bars: int         # e.g. 45
    gold_use_ema_gate: bool = False # If True, only buy Gold when Gold > 50 EMA, else hold 100% Cash


@dataclass
class SimResult:
    config: SimConfig
    initial_capital: float
    final_capital: float
    total_trades: int
    gold_trades: int
    stock_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    net_pnl: float
    net_pnl_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    total_taxes: float
    trades: List[SimTrade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)


def run_orchestrator_simulation(
    df_raw: pd.DataFrame,
    config: SimConfig,
    initial_capital: float = INITIAL_CAPITAL_INR,
    warmup_bars: int = 120,
) -> SimResult:
    """Executes a day-by-day simulated replay of the live orchestrator logic."""
    
    # 1. Prepare indicator DataFrames for all symbols
    all_syms = list(set([BENCHMARK_SYMBOL, SAFE_ASSET_SYMBOL] + ETFS_ALL + STOCKS_ALL))
    data_map: Dict[str, pd.DataFrame] = {}
    
    for sym in all_syms:
        try:
            sub = pd.DataFrame(index=df_raw.index)
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                val = df_raw[col][sym] if isinstance(df_raw.columns, pd.MultiIndex) else df_raw[col]
                sub[col.lower()] = val.astype(float)
            sub = add_all_indicators(sub.ffill().dropna())
            if len(sub) > warmup_bars:
                data_map[sym] = sub
        except Exception:
            pass

    if BENCHMARK_SYMBOL not in data_map:
        raise ValueError("Benchmark data missing.")

    df_bm = data_map[BENCHMARK_SYMBOL]
    dates = df_bm.index[warmup_bars:]

    capital = initial_capital
    open_positions: Dict[str, Dict[str, Any]] = {}
    closed_trades: List[SimTrade] = []
    equity_curve_vals = []
    total_taxes = 0.0

    for idx, dt in enumerate(dates):
        d_str = dt.strftime("%Y-%m-%d")
        t_global = df_bm.index.get_loc(dt)

        # ── Step 1: Detect Macro Regime ──
        bm_bar = df_bm.loc[dt]
        bm_px = float(bm_bar["close"])
        bm_sma200 = float(bm_bar.get("sma_200", bm_px))
        bm_ema12 = float(bm_bar.get("ema_12", bm_px))
        bm_ema50 = float(bm_bar.get("ema_50", bm_px))
        bm_adx = float(bm_bar.get("adx", 20.0))

        if bm_px <= bm_sma200 or (bm_ema12 < bm_ema50 * 0.99):
            regime = "BEAR_DEFENSE"
        elif bm_adx >= 22.0 and bm_ema12 > bm_ema50:
            regime = "TRENDING_BULL"
        else:
            regime = "CHOPPY_SIDEWAYS"

        # ── Step 2: Manage Open Positions ──
        for sym in list(open_positions.keys()):
            pos = open_positions[sym]
            df_s = data_map.get(sym)
            if df_s is None or dt not in df_s.index:
                continue

            bar = df_s.loc[dt]
            curr_px = float(bar["close"])
            low_px = float(bar["low"])
            high_px = float(bar["high"])
            entry_px = pos["entry_price"]
            sl_px = pos["stop_loss"]
            tp_px = pos["take_profit"]
            qty = pos["quantity"]
            risk_unit = pos["risk_unit"]
            bars_held = idx - pos["entry_bar_idx"]

            # Break-Even Lock Check
            if not pos["be_locked"] and high_px >= (entry_px + config.be_lock_r * risk_unit):
                pos["stop_loss"] = entry_px
                pos["be_locked"] = True
                sl_px = entry_px

            exit_triggered = False
            exit_px = curr_px
            exit_reason = ""

            # Check SL
            if low_px <= sl_px:
                exit_triggered = True
                exit_px = sl_px
                exit_reason = "STOP_LOSS"
            # Check TP
            elif high_px >= tp_px:
                exit_triggered = True
                exit_px = tp_px
                exit_reason = "TAKE_PROFIT"
            # Check Bear Rotation (Liquidate equities if market flips to Bear)
            elif regime == "BEAR_DEFENSE" and sym != SAFE_ASSET_SYMBOL:
                exit_triggered = True
                exit_px = curr_px
                exit_reason = "BEAR_ROTATION"
            # Check Time Exit
            elif bars_held >= config.max_hold_bars and sym != SAFE_ASSET_SYMBOL:
                exit_triggered = True
                exit_px = curr_px
                exit_reason = "TIME_EXIT"

            if exit_triggered:
                b_c, s_c = calculate_round_trip_cost(entry_px, exit_px, qty, "CNC")
                tax = b_c.total + s_c.total
                gross_sale = exit_px * qty
                gross_pnl = (exit_px - entry_px) * qty
                net_pnl = gross_pnl - tax

                capital += gross_sale - s_c.total
                total_taxes += tax
                open_positions.pop(sym)

                closed_trades.append(SimTrade(
                    symbol=sym,
                    strategy=pos["strategy"],
                    entry_date=pos["entry_date"],
                    exit_date=d_str,
                    entry_price=round(entry_px, 2),
                    exit_price=round(exit_px, 2),
                    quantity=qty,
                    gross_pnl=round(gross_pnl, 2),
                    taxes=round(tax, 2),
                    net_pnl=round(net_pnl, 2),
                    net_pnl_pct=round((net_pnl / (entry_px * qty)) * 100.0, 2),
                    exit_reason=exit_reason,
                    bars_held=bars_held,
                ))

        # ── Step 3: Scan & Enter New Positions ──
        available_slots = 3 - len(open_positions)

        if regime == "BEAR_DEFENSE":
            if SAFE_ASSET_SYMBOL not in open_positions and capital > 2000:
                df_gold = data_map.get(SAFE_ASSET_SYMBOL)
                if df_gold is not None and dt in df_gold.index:
                    pos_g = df_gold.index.get_loc(dt)
                    g_px = float(df_gold.loc[dt]["close"])
                    g_ema50 = float(df_gold["close"].iloc[:pos_g+1].ewm(span=50, adjust=False).mean().iloc[-1]) if pos_g >= 50 else g_px

                    # If Gold 50-EMA Trend Gate is enabled, only enter if Gold > 50 EMA
                    allow_gold_entry = (g_px > g_ema50) if config.gold_use_ema_gate else True

                    if allow_gold_entry:
                        g_qty = math.floor((capital * 0.95) / g_px)
                        if g_qty > 0:
                            b_c, _ = calculate_round_trip_cost(g_px, g_px, g_qty, "CNC")
                            cost = (g_qty * g_px) + b_c.total
                            if capital >= cost:
                                capital -= cost
                                total_taxes += b_c.total
                                sl = round(g_px * (1.0 - config.gold_sl_pct), 2)
                                tp = round(g_px * (1.0 + config.gold_tp_pct), 2)
                                open_positions[SAFE_ASSET_SYMBOL] = {
                                    "strategy": "Sovereign Gold Defense",
                                    "entry_date": d_str,
                                    "entry_price": g_px,
                                    "quantity": g_qty,
                                    "stop_loss": sl,
                                    "take_profit": tp,
                                    "risk_unit": round(g_px * config.gold_sl_pct, 2),
                                    "be_locked": False,
                                    "entry_bar_idx": idx,
                                }

        elif available_slots > 0 and capital > 5000:
            alloc_per_slot = capital / available_slots
            candidates = []

            for sym in STOCKS_ALL:
                if sym in open_positions:
                    continue
                df_s = data_map.get(sym)
                if df_s is None or dt not in df_s.index or t_global < 60:
                    continue

                pos_s = df_s.index.get_loc(dt)
                if pos_s < 60:
                    continue

                bar = df_s.iloc[pos_s]
                prev_bar = df_s.iloc[pos_s - 1]
                px = float(bar["close"])
                prev_px = float(prev_bar["close"])
                sma20 = float(bar.get("sma_20", px))
                prev_sma20 = float(prev_bar.get("sma_20", prev_px))
                rsi = float(bar.get("rsi", 50.0))
                atr = float(bar.get("atr", px * 0.02))

                # 60d RS Gate
                rs_today = px / bm_px
                rs_60 = float(df_s.iloc[pos_s - 60]["close"]) / float(df_bm.iloc[t_global - 60]["close"])
                rs_slope = ((rs_today - rs_60) / rs_60) * 100.0

                is_pullback = (prev_px <= prev_sma20 * 1.008) and (px > sma20) and (40.0 <= rsi <= 60.0) and (rs_slope > 0)
                if is_pullback:
                    candidates.append((sym, "Large-Cap RS Pullback", px, atr, rs_slope))

            candidates.sort(key=lambda x: x[4], reverse=True)

            for sym, strat, px, atr, rs in candidates[:available_slots]:
                qty = math.floor(alloc_per_slot / (px * 1.005))
                if qty > 0:
                    b_c, _ = calculate_round_trip_cost(px, px, qty, "CNC")
                    order_cost = (qty * px) + b_c.total
                    if capital >= order_cost:
                        sl = round(px - 1.25 * atr, 2)
                        tp = round(px + config.pullback_tp_mult * atr, 2)
                        capital -= order_cost
                        total_taxes += b_c.total
                        open_positions[sym] = {
                            "strategy": strat,
                            "entry_date": d_str,
                            "entry_price": px,
                            "quantity": qty,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "risk_unit": round(1.25 * atr, 2),
                            "be_locked": False,
                            "entry_bar_idx": idx,
                        }

        # ── Step 4: Track Daily Mark-to-Market Equity ──
        open_mtm = 0.0
        for s, p in open_positions.items():
            df_s = data_map.get(s)
            if df_s is not None and dt in df_s.index:
                open_mtm += p["quantity"] * float(df_s.loc[dt]["close"])
            else:
                open_mtm += p["quantity"] * p["entry_price"]

        nav = capital + open_mtm
        equity_curve_vals.append(nav)

    # ── Final Liquidation of Open Positions ──
    final_dt = dates[-1]
    d_final_str = final_dt.strftime("%Y-%m-%d")
    for sym, pos in list(open_positions.items()):
        df_s = data_map.get(sym)
        px = float(df_s.loc[final_dt]["close"]) if df_s is not None and final_dt in df_s.index else pos["entry_price"]
        qty = pos["quantity"]
        entry_px = pos["entry_price"]
        b_c, s_c = calculate_round_trip_cost(entry_px, px, qty, "CNC")
        tax = b_c.total + s_c.total
        gross_sale = px * qty
        gross_pnl = (px - entry_px) * qty
        net_pnl = gross_pnl - tax

        capital += gross_sale - s_c.total
        total_taxes += tax
        closed_trades.append(SimTrade(
            symbol=sym,
            strategy=pos["strategy"],
            entry_date=pos["entry_date"],
            exit_date=d_final_str,
            entry_price=round(entry_px, 2),
            exit_price=round(px, 2),
            quantity=qty,
            gross_pnl=round(gross_pnl, 2),
            taxes=round(tax, 2),
            net_pnl=round(net_pnl, 2),
            net_pnl_pct=round((net_pnl / (entry_px * qty)) * 100.0, 2),
            exit_reason="SIMULATION_END",
            bars_held=len(dates) - 1 - pos["entry_bar_idx"],
        ))

    equity_series = pd.Series(equity_curve_vals, index=dates)
    final_cap = float(equity_series.iloc[-1])
    net_pnl = final_cap - initial_capital
    net_pct = (net_pnl / initial_capital) * 100.0

    years = len(dates) / 252.0 if len(dates) > 0 else 1.0
    cagr = ((final_cap / initial_capital) ** (1.0 / years) - 1.0) * 100.0 if years > 0 and final_cap > 0 else 0.0

    peak = equity_series.cummax()
    dd = (peak - equity_series) / peak * 100.0
    max_dd = float(dd.max()) if not dd.empty else 0.0

    daily_rets = equity_series.pct_change().dropna()
    if len(daily_rets) > 1 and daily_rets.std() > 0:
        sharpe = float((daily_rets.mean() / daily_rets.std()) * np.sqrt(252))
        neg_rets = daily_rets[daily_rets < 0]
        sortino = float((daily_rets.mean() / neg_rets.std()) * np.sqrt(252)) if len(neg_rets) > 1 and neg_rets.std() > 0 else sharpe
    else:
        sharpe, sortino = 0.0, 0.0

    wins = [t for t in closed_trades if t.net_pnl > 0]
    losses = [t for t in closed_trades if t.net_pnl <= 0]
    win_rate = (len(wins) / len(closed_trades) * 100.0) if closed_trades else 0.0

    gross_profit = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss else float("inf")

    gold_trades = sum(1 for t in closed_trades if t.symbol == SAFE_ASSET_SYMBOL)
    stock_trades = len(closed_trades) - gold_trades

    return SimResult(
        config=config,
        initial_capital=initial_capital,
        final_capital=round(final_cap, 2),
        total_trades=len(closed_trades),
        gold_trades=gold_trades,
        stock_trades=stock_trades,
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=round(win_rate, 1),
        profit_factor=profit_factor,
        net_pnl=round(net_pnl, 2),
        net_pnl_pct=round(net_pct, 2),
        cagr_pct=round(cagr, 2),
        max_drawdown_pct=round(max_dd, 2),
        sharpe_ratio=round(sharpe, 2),
        sortino_ratio=round(sortino, 2),
        total_taxes=round(total_taxes, 2),
        trades=closed_trades,
        equity_curve=equity_series,
    )


def main():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🔬 LIVE PAPER ORCHESTRATOR: GOLD 50-EMA TREND GATE & SL/TP CALIBRATION[/]\n"
        "[dim]Benchmarking 8% SL, 15% TP & Gold 50-EMA Trend Gate on Real NSE Data[/]",
        border_style="cyan"
    ))
    console.print()

    # Ingest 5-Year Data
    all_syms = list(set([BENCHMARK_SYMBOL, SAFE_ASSET_SYMBOL] + ETFS_ALL + STOCKS_ALL))
    console.print(f"[dim cyan]Downloading 5-Year NSE Daily Market Feeds ({len(all_syms)} instruments)...[/]")
    df_raw = yf.download(all_syms, period="5y", interval="1d", progress=False)

    configs = [
        SimConfig(name="Baseline (Old Live: 5% SL / 25% TP)", gold_sl_pct=0.05, gold_tp_pct=0.25, pullback_tp_mult=3.5, be_lock_r=1.5, max_hold_bars=999, gold_use_ema_gate=False),
        SimConfig(name="Calibrated (8% SL / 15% TP)", gold_sl_pct=0.08, gold_tp_pct=0.15, pullback_tp_mult=4.0, be_lock_r=2.0, max_hold_bars=45, gold_use_ema_gate=False),
        SimConfig(name="Upgraded Shield (8% SL / 15% TP + 50-EMA Gate)", gold_sl_pct=0.08, gold_tp_pct=0.15, pullback_tp_mult=4.0, be_lock_r=2.0, max_hold_bars=45, gold_use_ema_gate=True),
    ]

    # 1. 2021–2024 Stress Test
    df_2021_2024 = df_raw.loc[:"2024-12-31"]
    console.print(f"[bold yellow]Running 2021–2024 Stress Test ({len(df_2021_2024)} bars)...[/]")
    results_2021_2024 = [run_orchestrator_simulation(df_2021_2024, cfg, warmup_bars=120) for cfg in configs]

    # 2. 2025–2026 Forward Test
    df_fwd = df_raw.iloc[-350:]
    console.print(f"[bold yellow]Running 2025–2026 Forward Test ({len(df_fwd)} bars)...[/]")
    results_fwd = [run_orchestrator_simulation(df_fwd, cfg, warmup_bars=100) for cfg in configs]

    # 3. Full 5-Year Combined Replay
    console.print(f"[bold yellow]Running Full 5-Year Replay ({len(df_raw)} bars)...[/]")
    results_5y = [run_orchestrator_simulation(df_raw, cfg, warmup_bars=120) for cfg in configs]

    def print_tbl(title, results):
        tbl = Table(title=title, box=box.DOUBLE_EDGE, header_style="bold cyan")
        tbl.add_column("Configuration", style="bold", width=44)
        tbl.add_column("Final NAV (₹)", justify="right", width=16)
        tbl.add_column("Net Return (%)", justify="right", width=16)
        tbl.add_column("CAGR (%)", justify="right", width=12)
        tbl.add_column("Max DD (%)", justify="right", width=12)
        tbl.add_column("Win Rate", justify="right", width=10)
        tbl.add_column("Profit Factor", justify="right", width=14)
        tbl.add_column("Gold Trades", justify="center", width=12)
        tbl.add_column("Taxes (₹)", justify="right", width=12)

        for r in results:
            pnl_color = "green" if r.net_pnl >= 0 else "red"
            tbl.add_row(
                r.config.name,
                f"₹{r.final_capital:,.2f}",
                f"[{pnl_color}]+{r.net_pnl_pct:.2f}%[/]" if r.net_pnl >= 0 else f"[{pnl_color}]{r.net_pnl_pct:.2f}%[/]",
                f"[{pnl_color}]{r.cagr_pct:.2f}%[/]",
                f"[red]-{r.max_drawdown_pct:.2f}%[/]",
                f"{r.win_rate:.1f}%",
                f"{r.profit_factor:.2f}",
                f"{r.gold_trades} trades",
                f"₹{r.total_taxes:,.0f}",
            )
        console.print()
        console.print(tbl)
        console.print()

    print_tbl("[bold green]📊 1. 2021–2024 HISTORICAL STRESS TEST (NON-GOLD RALLY ERA)[/]", results_2021_2024)
    print_tbl("[bold green]📊 2. 2025–2026 FORWARD TEST (OUT-OF-SAMPLE)[/]", results_fwd)
    print_tbl("[bold green]📊 3. FULL 5-YEAR COMBINED REPLAY (2021–2026 MASTER)[/]", results_5y)




if __name__ == "__main__":
    main()
