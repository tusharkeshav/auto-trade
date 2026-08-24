# ─────────────────────────────────────────────────────────────────
#  run_2025_2026_granular_audit.py
#  Granular 2025 – 2026 Out-of-Sample Forward Test Audit.
# ─────────────────────────────────────────────────────────────────

import sys
from pathlib import Path
from typing import Dict, Any, List

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
from engine.ai_meta_orchestrator import AIMetaOrchestrator

console = Console()

START_DATE = pd.to_datetime("2025-01-01")
END_DATE   = pd.to_datetime("2026-08-23")


def analyze_forward_slice(equity_curve: pd.Series, trades: list, start_date=START_DATE, end_date=END_DATE, base_cap: float = 100000.0) -> Dict[str, Any]:
    eq = equity_curve.copy()
    eq.index = pd.to_datetime(eq.index).tz_localize(None)

    eq_slice = eq[(eq.index >= start_date) & (eq.index <= end_date)]
    if eq_slice.empty or len(eq_slice) < 2:
        return {}

    initial_val = float(eq_slice.iloc[0])
    scale_factor = base_cap / initial_val if initial_val > 0 else 1.0
    norm_eq = eq_slice * scale_factor

    final_val = float(norm_eq.iloc[-1])
    net_pnl   = final_val - base_cap
    net_pct   = (net_pnl / base_cap) * 100.0

    peak   = norm_eq.cummax()
    dd     = (peak - norm_eq) / peak * 100.0
    max_dd = float(dd.max()) if not dd.empty else 0.0

    daily_rets = norm_eq.pct_change().dropna()
    if len(daily_rets) > 1 and daily_rets.std() > 0:
        sharpe = float((daily_rets.mean() / daily_rets.std()) * np.sqrt(252))
        neg_rets = daily_rets[daily_rets < 0]
        sortino = float((daily_rets.mean() / neg_rets.std()) * np.sqrt(252)) if len(neg_rets) > 1 and neg_rets.std() > 0 else sharpe
    else:
        sharpe, sortino = 0.0, 0.0

    # Trades in 2025-2026
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

    # Max Losing Streak in 2025-2026
    max_loss_streak = 0
    curr_loss_streak = 0
    for t in slice_trades:
        if t.net_pnl <= 0:
            curr_loss_streak += 1
            if curr_loss_streak > max_loss_streak:
                max_loss_streak = curr_loss_streak
        else:
            curr_loss_streak = 0

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
        "loss_streak": max_loss_streak,
        "profit_factor": round(pf, 2),
        "taxes": round(taxes, 2),
    }


def main():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🔍 2025 – 2026 OUT-OF-SAMPLE FORWARD AUDIT: ALL STRATEGIES & AI ORCHESTRATOR[/]\n"
        "[dim]Auditing Pure Forward Performance in Choppy Sideways Market (2025-01-01 to Present)[/]",
        border_style="cyan"
    ))
    console.print()

    # Benchmark: NIFTY 50
    df_nifty = yf.download("^NSEI", start="2025-01-01", interval="1d", progress=False)["Close"].dropna()
    if isinstance(df_nifty.columns, pd.MultiIndex):
        df_nifty.columns = df_nifty.columns.get_level_values(0)
    nifty_start = float(df_nifty.iloc[0].item()) if hasattr(df_nifty.iloc[0], "item") else float(df_nifty.iloc[0])
    nifty_end   = float(df_nifty.iloc[-1].item()) if hasattr(df_nifty.iloc[-1], "item") else float(df_nifty.iloc[-1])
    nifty_ret   = ((nifty_end - nifty_start) / nifty_start) * 100.0

    console.print(f"[bold yellow]NIFTY 50 Benchmark Return (2025–2026): [green]+{nifty_ret:.2f}%[/] ({nifty_start:,.0f} -> {nifty_end:,.0f})[/]\n")

    results = {}

    console.print("[dim]1/7 Running Sector Rotation Engine...[/]")
    res_sec = SectorRotationEngine(initial_capital=100000.0, top_k=2, rebalance_interval=10).run(bars=1250)
    results["1. Sector Rotation"] = analyze_forward_slice(res_sec.equity_curve, res_sec.trades, base_cap=100000.0)

    console.print("[dim]2/7 Running Large-Cap Pullback Engine...[/]")
    res_pb = LargeCapPullbackEngine(capital=100000.0, max_open_trades=6, use_rs_gate=True).run(bars=1250)
    results["2. Large-Cap Pullback"] = analyze_forward_slice(res_pb.equity_curve, res_pb.trades, base_cap=100000.0)

    console.print("[dim]3/7 Running Minervini VCP Breakout...[/]")
    res_vcp = VCPBreakoutEngine(capital=100000.0, max_open_trades=6).run(bars=1250)
    results["3. Minervini VCP Breakout"] = analyze_forward_slice(res_vcp.equity_curve, res_vcp.trades, base_cap=100000.0)

    console.print("[dim]4/7 Running All-Weather Dual Book...[/]")
    res_db = AllWeatherDualBookEngine(total_capital=100000.0).run(bars=1250)
    all_db_trades = res_sec.trades + res_pb.trades
    results["4. All-Weather Dual Book"] = analyze_forward_slice(res_db.equity_curve, all_db_trades, base_cap=100000.0)

    console.print("[dim]5/7 Running Smart Dynamic Allocator...[/]")
    res_dyn = DynamicRegimeAllocatorEngine(total_capital=100000.0, adx_threshold=22.0).run(bars=1250)
    results["5. Smart Dynamic Allocator"] = analyze_forward_slice(res_dyn.equity_curve, all_db_trades, base_cap=100000.0)

    console.print("[dim]6/7 Running Smart Money Concepts (SMC)...[/]")
    res_smc = SMCLiquidityEngine(capital=100000.0, max_open_trades=6).run(bars=1250)
    results["6. Smart Money Concepts (SMC)"] = analyze_forward_slice(res_smc.equity_curve, res_smc.trades, base_cap=100000.0)

    console.print("[dim]7/7 Running AI Master Meta-Orchestrator...[/]")
    orch = AIMetaOrchestrator(total_capital=100000.0)
    res_orch = orch.run_backtest(bars=1250)
    all_orch_trades = res_sec.trades + res_pb.trades + res_vcp.trades + res_smc.trades
    all_orch_trades.sort(key=lambda x: x.exit_date)
    results["🏆 7. AI Meta-Orchestrator"] = analyze_forward_slice(res_orch.equity_curve, all_orch_trades, base_cap=100000.0)

    console.print()

    tbl = Table(title="[bold green]📊 2025 – 2026 FORWARD AUDIT SCORECARD (₹100,000 BASE)[/]", box=box.DOUBLE_EDGE, header_style="bold cyan")
    tbl.add_column("Strategy Engine", style="bold", width=28)
    tbl.add_column("Net P&L (₹)", justify="right", width=16)
    tbl.add_column("Return (%)", justify="right", width=14)
    tbl.add_column("Max DD", justify="right", width=12)
    tbl.add_column("Win Rate", justify="right", width=11)
    tbl.add_column("Max Loss Streak", justify="right", width=16)
    tbl.add_column("Profit Factor", justify="right", width=14)
    tbl.add_column("Taxes Paid", justify="right", width=14)

    for name, m in results.items():
        if not m: continue
        pnl_str = f"[bold green]+₹{m['net_pnl']:,.2f}[/]" if m['net_pnl'] >= 0 else f"[bold red]-₹{abs(m['net_pnl']):,.2f}[/]"
        ret_str = f"[bold green]+{m['net_pct']:.2f}%[/]" if m['net_pct'] >= 0 else f"[bold red]{m['net_pct']:.2f}%[/]"
        dd_str  = f"[bold green]-{m['max_dd']:.2f}%[/]" if m['max_dd'] < 10 else f"[bold yellow]-{m['max_dd']:.2f}%[/]"
        pf_str  = f"[bold yellow]{m['profit_factor']:.2f}[/]" if m['profit_factor'] < 99 else "∞"
        streak_str = f"[bold green]{m['loss_streak']} trades[/]" if m['loss_streak'] <= 3 else f"[yellow]{m['loss_streak']} trades[/]"

        tbl.add_row(
            name,
            pnl_str,
            ret_str,
            dd_str,
            f"{m['win_rate']:.1f}%",
            streak_str,
            pf_str,
            f"₹{m['taxes']:,.2f}",
        )

    console.print(tbl)
    console.print()


if __name__ == "__main__":
    main()
