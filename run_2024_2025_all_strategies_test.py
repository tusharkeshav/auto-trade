# ─────────────────────────────────────────────────────────────────
#  run_2024_2025_all_strategies_test.py
#  Master Multi-Strategy Benchmark for Calendar Year 2024 – 2025.
# ─────────────────────────────────────────────────────────────────

import sys
from pathlib import Path
from typing import Dict, Any

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

from strategies.sector_rotation.engine import SectorRotationEngine
from strategies.largecap_pullback.engine import LargeCapPullbackEngine
from strategies.vcp_breakout.engine import VCPBreakoutEngine
from strategies.all_weather_dual_book.engine import AllWeatherDualBookEngine
from strategies.smart_dynamic_regime.engine import DynamicRegimeAllocatorEngine
from strategies.smc_liquidity_engine.engine import SMCLiquidityEngine

console = Console()

START_DATE = pd.to_datetime("2024-01-01")
END_DATE   = pd.to_datetime("2025-01-01")


def slice_metrics(equity_curve: pd.Series, trades: list, start_date=START_DATE, end_date=END_DATE, base_cap: float = 100000.0) -> Dict[str, Any]:
    # Normalize index to timezone-naive datetimes
    eq = equity_curve.copy()
    eq.index = pd.to_datetime(eq.index).tz_localize(None)

    eq_slice = eq[(eq.index >= start_date) & (eq.index <= end_date)]
    if eq_slice.empty or len(eq_slice) < 2:
        return {}

    # Normalize starting capital to base_cap at start_date
    initial_val = float(eq_slice.iloc[0])
    scale_factor = base_cap / initial_val if initial_val > 0 else 1.0
    norm_eq = eq_slice * scale_factor

    final_val = float(norm_eq.iloc[-1])
    net_pnl   = final_val - base_cap
    net_pct   = (net_pnl / base_cap) * 100.0

    # Max Drawdown in 2024
    peak   = norm_eq.cummax()
    dd     = (peak - norm_eq) / peak * 100.0
    max_dd = float(dd.max()) if not dd.empty else 0.0

    # Daily Sharpe and Sortino in 2024
    daily_rets = norm_eq.pct_change().dropna()
    if len(daily_rets) > 1 and daily_rets.std() > 0:
        sharpe = float((daily_rets.mean() / daily_rets.std()) * np.sqrt(252))
        neg_rets = daily_rets[daily_rets < 0]
        sortino = float((daily_rets.mean() / neg_rets.std()) * np.sqrt(252)) if len(neg_rets) > 1 and neg_rets.std() > 0 else sharpe
    else:
        sharpe, sortino = 0.0, 0.0

    # Trades in 2024
    slice_trades = []
    for t in trades:
        t_exit = pd.to_datetime(t.exit_date).tz_localize(None) if hasattr(t.exit_date, 'tzinfo') and t.exit_date.tzinfo else pd.to_datetime(t.exit_date)
        if start_date <= t_exit <= end_date:
            slice_trades.append(t)

    n_trades = len(slice_trades)
    wins   = [t for t in slice_trades if t.net_pnl > 0]
    losses = [t for t in slice_trades if t.net_pnl <= 0]
    win_rate = (len(wins) / n_trades * 100.0) if n_trades else 0.0
    gross_win = sum(t.net_pnl for t in wins)
    gross_loss= abs(sum(t.net_pnl for t in losses))
    pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    taxes = sum(t.cost_inr for t in slice_trades) * scale_factor

    return {
        "base_capital": base_cap,
        "final_capital": round(final_val, 2),
        "net_pnl": round(net_pnl, 2),
        "net_pct": round(net_pct, 2),
        "max_dd": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "trades": n_trades,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "taxes": round(taxes, 2),
    }


def main():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🗓️ 2024 – 2025 OUT-OF-SAMPLE BENCHMARK: ALL 6 QUANTITATIVE STRATEGIES[/]\n"
        "[dim]Auditing Pure Forward Performance Across Calendar Year 2024 (2024-01-01 to 2025-01-01)[/]",
        border_style="cyan"
    ))
    console.print()

    # Benchmark: NIFTY 50
    df_nifty = yf.download("^NSEI", start="2024-01-01", end="2025-01-05", interval="1d", progress=False)["Close"]
    if isinstance(df_nifty.columns, pd.MultiIndex):
        df_nifty.columns = df_nifty.columns.get_level_values(0)
    nifty_start = float(df_nifty.iloc[0])
    nifty_end   = float(df_nifty.iloc[-1])
    nifty_ret   = ((nifty_end - nifty_start) / nifty_start) * 100.0

    console.print(f"[bold yellow]NIFTY 50 Benchmark Return in 2024: [green]+{nifty_ret:.2f}%[/] ({nifty_start:,.0f} -> {nifty_end:,.0f})[/]\n")

    results = {}

    console.print("[dim]1/6 Running Sector Rotation Engine...[/]")
    res_sec = SectorRotationEngine(initial_capital=100000.0, top_k=2, rebalance_interval=10).run(bars=1250)
    results["1. Sector Rotation"] = slice_metrics(res_sec.equity_curve, res_sec.trades, base_cap=100000.0)

    console.print("[dim]2/6 Running Large-Cap Pullback Engine...[/]")
    res_pb = LargeCapPullbackEngine(capital=100000.0, max_open_trades=6, use_rs_gate=True).run(bars=1250)
    results["2. Large-Cap Pullback"] = slice_metrics(res_pb.equity_curve, res_pb.trades, base_cap=100000.0)

    console.print("[dim]3/6 Running Minervini VCP Breakout...[/]")
    res_vcp = VCPBreakoutEngine(capital=100000.0, max_open_trades=6).run(bars=1250)
    results["3. Minervini VCP"] = slice_metrics(res_vcp.equity_curve, res_vcp.trades, base_cap=100000.0)

    console.print("[dim]4/6 Running All-Weather Dual Book...[/]")
    res_db = AllWeatherDualBookEngine(total_capital=100000.0).run(bars=1250)
    all_db_trades = res_sec.trades + res_pb.trades
    results["4. All-Weather 50/50 Dual Book"] = slice_metrics(res_db.equity_curve, all_db_trades, base_cap=100000.0)

    console.print("[dim]5/6 Running Smart Dynamic Regime Allocator...[/]")
    res_dyn = DynamicRegimeAllocatorEngine(total_capital=100000.0, adx_threshold=22.0).run(bars=1250)
    results["5. 🏆 Smart Dynamic Allocator"] = slice_metrics(res_dyn.equity_curve, all_db_trades, base_cap=100000.0)

    console.print("[dim]6/6 Running Smart Money Concepts (SMC)...[/]")
    res_smc = SMCLiquidityEngine(capital=100000.0, max_open_trades=6).run(bars=1250)
    results["6. Smart Money Concepts (SMC)"] = slice_metrics(res_smc.equity_curve, res_smc.trades, base_cap=100000.0)

    console.print()

    tbl = Table(title="[bold green]📊 2024 – 2025 MASTER COMPARATIVE SCORECARD (₹100,000 BASE)[/]", box=box.DOUBLE_EDGE, header_style="bold cyan")
    tbl.add_column("Strategy Name", style="bold", width=30)
    tbl.add_column("2024 Net P&L (₹)", justify="right", width=18)
    tbl.add_column("2024 Return (%)", justify="right", width=18)
    tbl.add_column("Max DD (%)", justify="right", width=13)
    tbl.add_column("Win Rate", justify="right", width=12)
    tbl.add_column("Profit Factor", justify="right", width=14)
    tbl.add_column("Sortino", justify="right", width=10)
    tbl.add_column("Trades", justify="right", width=10)

    for name, m in results.items():
        if not m: continue
        pnl_str = f"[bold green]+₹{m['net_pnl']:,.2f}[/]" if m['net_pnl'] >= 0 else f"[bold red]-₹{abs(m['net_pnl']):,.2f}[/]"
        ret_str = f"[bold green]+{m['net_pct']:.2f}%[/]" if m['net_pct'] >= 0 else f"[bold red]{m['net_pct']:.2f}%[/]"
        dd_str  = f"[bold green]-{m['max_dd']:.2f}%[/]" if m['max_dd'] < 10 else f"[bold yellow]-{m['max_dd']:.2f}%[/]"
        pf_str  = f"[bold yellow]{m['profit_factor']:.2f}[/]" if m['profit_factor'] < 99 else "∞"
        tbl.add_row(
            name,
            pnl_str,
            ret_str,
            dd_str,
            f"{m['win_rate']:.1f}%",
            pf_str,
            f"{m['sortino']:.2f}",
            f"{m['trades']}",
        )

    console.print(tbl)
    console.print()


if __name__ == "__main__":
    main()
