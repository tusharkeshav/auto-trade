# ─────────────────────────────────────────────────────────────────
#  run_etf_isolation_experiment.py  —  2×2 Scientific Quant Experiment
#
#  Objective:
#    Isolate and quantify the exact alpha destruction caused by:
#      1. Indicator Noise (Spot Index signals vs. ETF share signals)
#      2. Transaction Cost Drag (Zero Cost vs. Real Indian CNC Delivery Tax)
#
#  Experimental Matrix (Executed strictly on BANKBEES.NS Cash ETF):
#    - Exp A (Zero Cost) : Signal from BANKNIFTY Index  → Execute on BANKBEES (0 Fees)
#    - Exp A (Real Cost) : Signal from BANKNIFTY Index  → Execute on BANKBEES (Real CNC Tax)
#    - Exp B (Zero Cost) : Signal from BANKBEES ETF     → Execute on BANKBEES (0 Fees)
#    - Exp B (Real Cost) : Signal from BANKBEES ETF     → Execute on BANKBEES (Real CNC Tax)
#
#  Usage:
#      python run_etf_isolation_experiment.py
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.india_settings import INITIAL_CAPITAL_INR, INDIA_MAX_RISK_PER_TRADE_PCT, INDIA_SIGNAL_THRESHOLD
from data.india.nse_client import NSEClient
from indicators import add_all_indicators
from probability.unified_cross_scorer import UnifiedCrossScorer
from engine.india_costs import calculate_round_trip_cost

console = Console()


@dataclass
class IsoTrade:
    entry_time:   datetime
    exit_time:    datetime
    direction:    str
    entry_price:  float
    exit_price:   float
    qty:          float
    gross_pnl:    float
    cost_inr:     float
    net_pnl:      float
    net_pnl_pct:  float
    exit_type:    str
    setup_label:  str


@dataclass
class IsoResult:
    label:           str
    signal_source:   str
    cost_model:      str
    trades:          list[IsoTrade] = field(default_factory=list)
    initial_capital: float = INITIAL_CAPITAL_INR
    final_capital:   float = INITIAL_CAPITAL_INR

    @property
    def total_trades(self) -> int: return len(self.trades)

    @property
    def winning_trades(self) -> list[IsoTrade]: return [t for t in self.trades if t.net_pnl > 0]

    @property
    def losing_trades(self) -> list[IsoTrade]: return [t for t in self.trades if t.net_pnl <= 0]

    @property
    def win_rate(self) -> float: return len(self.winning_trades) / self.total_trades * 100 if self.trades else 0.0

    @property
    def total_net_pnl(self) -> float: return self.final_capital - self.initial_capital

    @property
    def total_net_pnl_pct(self) -> float: return self.total_net_pnl / self.initial_capital * 100

    @property
    def total_gross_pnl(self) -> float: return sum(t.gross_pnl for t in self.trades)

    @property
    def total_costs(self) -> float: return sum(t.cost_inr for t in self.trades)

    @property
    def profit_factor(self) -> float:
        gp = sum(t.net_pnl for t in self.winning_trades)
        gl = abs(sum(t.net_pnl for t in self.losing_trades))
        return round(gp / gl, 2) if gl else float("inf")


class IsolationEngine:
    """
    Executes a specific quadrant of the 2×2 Isolation Experiment.
    """

    def __init__(
        self,
        label:         str,
        signal_source: str,   # "BANKNIFTY" or "BANKBEES.NS"
        zero_cost:     bool,  # True = ₹0 cost, False = Real CNC Tax
        bars:          int = 1400,
        interval:      str = "1h",
        vix:           float = 16.0,
    ):
        self.label         = label
        self.signal_source = signal_source
        self.zero_cost     = zero_cost
        self.bars          = bars
        self.interval      = interval
        self.vix           = vix
        self.client        = NSEClient()
        self.scorer        = UnifiedCrossScorer(
            symbol      = signal_source,
            threshold   = INDIA_SIGNAL_THRESHOLD,
            atr_sl_mult = 1.0,
            atr_tp_mult = 2.5,
            interval    = interval,
            current_vix = vix,
        )

    def run(self, df_sig: pd.DataFrame, df_exe: pd.DataFrame) -> IsoResult:
        """
        Run simulation: generate signals from df_sig, execute trades on df_exe.
        """
        WARMUP = min(200, len(df_sig) // 4)
        result = IsoResult(
            label         = self.label,
            signal_source = self.signal_source,
            cost_model    = "Zero Cost" if self.zero_cost else "Real CNC Tax",
            initial_capital = INITIAL_CAPITAL_INR,
            final_capital   = INITIAL_CAPITAL_INR,
        )
        cash = INITIAL_CAPITAL_INR
        consec_losses = 0
        cooldown_until = -1

        i = WARMUP
        while i < len(df_sig) - 1:
            if i < cooldown_until:
                i += 1
                continue

            row_sig = df_sig.iloc[i]
            slice_sig = df_sig.iloc[: i + 1]

            signal = self.scorer.score(row_sig, slice_sig)

            # In Cash Delivery (CNC), overnight shorting is forbidden by SEBI
            if not signal.is_tradeable() or signal.direction == "SHORT":
                i += 1
                continue

            # Execute on ETF dataframe (df_exe) at index i
            row_exe = df_exe.iloc[i]
            entry_exe   = float(row_exe["close"])
            atr_sig     = row_sig.get("atr") if hasattr(row_sig, "get") else row_sig["atr"]
            if math.isnan(atr_sig) or atr_sig <= 0:
                atr_sig = signal.entry_price * 0.006

            # Scale SL distance proportionally from Index to ETF price
            price_ratio = entry_exe / float(row_sig["close"]) if float(row_sig["close"]) > 0 else 1.0
            sl_dist_exe = (atr_sig * self.scorer.atr_sl_mult) * price_ratio

            sl_exe = round(entry_exe - sl_dist_exe, 2)
            tp_exe = round(entry_exe + sl_dist_exe * self.scorer.atr_tp_mult, 2)

            risk_inr = cash * INDIA_MAX_RISK_PER_TRADE_PCT / 100
            qty      = risk_inr / sl_dist_exe if sl_dist_exe > 0 else 0
            # Cash cap: cannot buy on leverage in CNC Delivery
            if entry_exe > 0:
                qty = min(qty, cash / entry_exe)
            if qty <= 0:
                i += 1
                continue

            trade = self._simulate_forward(df_exe, signal, i, qty, entry_exe, sl_dist_exe, sl_exe, tp_exe)

            if trade:
                cash += trade.net_pnl
                result.trades.append(trade)

                if trade.exit_type in ("STOP_LOSS", "TRAIL_STOP") and trade.net_pnl <= 0:
                    consec_losses += 1
                    if consec_losses >= 4:
                        cooldown_until = i + 48
                        consec_losses = 0
                else:
                    consec_losses = 0

                i += (df_exe.index.get_loc(trade.exit_time) - i) if trade.exit_time in df_exe.index else 1
                continue

            i += 1

        result.final_capital = round(cash, 2)
        return result

    def _simulate_forward(
        self,
        df_exe:    pd.DataFrame,
        signal,
        entry_idx: int,
        qty:       float,
        entry_exe: float,
        sl_dist:   float,
        sl:        float,
        tp:        float,
    ) -> IsoTrade | None:
        direction = signal.direction
        timeout   = len(df_exe)

        for j in range(entry_idx + 1, len(df_exe)):
            candle    = df_exe.iloc[j]
            high, low = float(candle["high"]), float(candle["low"])
            ts        = df_exe.index[j].to_pydatetime()

            if direction == "LONG":
                if low <= sl:
                    exit_type = "TRAIL_STOP" if sl >= entry_exe else "STOP_LOSS"
                    return self._make_trade(signal, df_exe, entry_idx, j, entry_exe, sl, exit_type, qty)
                if high >= tp:
                    return self._make_trade(signal, df_exe, entry_idx, j, entry_exe, tp, "TAKE_PROFIT", qty)

                # Trailing stop
                if high >= entry_exe + sl_dist * 1.5 and sl < entry_exe + sl_dist * 0.5:
                    sl = round(entry_exe + sl_dist * 0.5, 2)
                elif high >= entry_exe + sl_dist * 1.0 and sl < entry_exe:
                    sl = round(entry_exe, 2)
            else:
                if high >= sl:
                    exit_type = "TRAIL_STOP" if sl <= entry_exe else "STOP_LOSS"
                    return self._make_trade(signal, df_exe, entry_idx, j, entry_exe, sl, exit_type, qty)
                if low <= tp:
                    return self._make_trade(signal, df_exe, entry_idx, j, entry_exe, tp, "TAKE_PROFIT", qty)

                if low <= entry_exe - sl_dist * 1.5 and sl > entry_exe - sl_dist * 0.5:
                    sl = round(entry_exe - sl_dist * 0.5, 2)
                elif low <= entry_exe - sl_dist * 1.0 and sl > entry_exe:
                    sl = round(entry_exe, 2)

        j = len(df_exe) - 1
        exit_price = float(df_exe.iloc[j]["close"])
        return self._make_trade(signal, df_exe, entry_idx, j, entry_exe, exit_price, "TIMEOUT", qty)

    def _make_trade(
        self,
        signal,
        df_exe:      pd.DataFrame,
        entry_idx:   int,
        exit_idx:    int,
        entry_price: float,
        exit_price:  float,
        exit_type:   str,
        qty:         float,
    ) -> IsoTrade:
        if signal.direction == "LONG":
            gross_pnl = (exit_price - entry_price) * qty
        else:
            gross_pnl = (entry_price - exit_price) * qty

        if self.zero_cost:
            cost_inr = 0.0
        else:
            buy_cost, sell_cost = calculate_round_trip_cost(entry_price, exit_price, qty, "CNC")
            cost_inr = buy_cost.total + sell_cost.total

        net_pnl = gross_pnl - cost_inr
        net_pct = net_pnl / (entry_price * qty) * 100 if (entry_price * qty) > 0 else 0.0

        reason_str = signal.reason or ""
        setup_lbl  = "DUAL_CONFIRM" if "DUAL CONFIRM" in reason_str else ("BAND_BOUNCE" if "BAND BOUNCE" in reason_str else "PULLBACK")

        return IsoTrade(
            entry_time  = df_exe.index[entry_idx].to_pydatetime(),
            exit_time   = df_exe.index[exit_idx].to_pydatetime(),
            direction   = signal.direction,
            entry_price = round(entry_price, 2),
            exit_price  = round(exit_price, 2),
            qty         = qty,
            gross_pnl   = round(gross_pnl, 2),
            cost_inr    = round(cost_inr, 2),
            net_pnl     = round(net_pnl, 2),
            net_pnl_pct = round(net_pct, 4),
            exit_type   = exit_type,
            setup_label = setup_lbl,
        )


def main() -> None:
    console.print("\n[bold cyan]── 2×2 SCIENTIFIC FACTOR ISOLATION EXPERIMENT (1h Timeframe) ──[/]")
    console.print("[dim]Target Asset: BANKBEES.NS Cash ETF (Unleveraged, CNC Delivery)[/]")
    console.print("[dim]Isolating: [1] Indicator Noise (Index vs ETF) | [2] Tax Drag (Zero vs Real Cost)[/]\n")

    client = NSEClient()
    logger.info("Fetching and aligning 1,400 hourly candles for BANKNIFTY Index & BANKBEES.NS ETF...")
    df_idx = add_all_indicators(client.get_ohlcv("BANKNIFTY", "1h", 1400))
    df_etf = add_all_indicators(client.get_ohlcv("BANKBEES.NS", "1h", 1400))

    # Align strictly on shared intersection timestamps
    common_idx = df_idx.index.intersection(df_etf.index)
    df_idx = df_idx.loc[common_idx]
    df_etf = df_etf.loc[common_idx]
    logger.info(f"Aligned on {len(common_idx)} common hourly candles ({common_idx[0].strftime('%Y-%m-%d')} → {common_idx[-1].strftime('%Y-%m-%d')}).\n")

    # ── Execute Experimental Matrix ──────────────────────────────────
    # Exp A1: Index Signal → ETF Execute (Zero Cost)
    exp_a1 = IsolationEngine("Exp A (Zero Cost)", "BANKNIFTY", zero_cost=True).run(df_idx, df_etf)
    # Exp A2: Index Signal → ETF Execute (Real Cost)
    exp_a2 = IsolationEngine("Exp A (Real Costs)", "BANKNIFTY", zero_cost=False).run(df_idx, df_etf)

    # Exp B1: ETF Signal   → ETF Execute (Zero Cost)
    exp_b1 = IsolationEngine("Exp B (Zero Cost)", "BANKBEES.NS", zero_cost=True).run(df_etf, df_etf)
    # Exp B2: ETF Signal   → ETF Execute (Real Cost)
    exp_b2 = IsolationEngine("Exp B (Real Costs)", "BANKBEES.NS", zero_cost=False).run(df_etf, df_etf)

    # ── Print Scientific Synthesis Table ─────────────────────────────
    tbl = Table(box=box.DOUBLE_EDGE, show_header=True, header_style="bold cyan", title="\n🏆 SCIENTIFIC ISOLATION RESULTS (BANKBEES.NS Cash ETF)")
    tbl.add_column("Experiment Label",      width=20)
    tbl.add_column("Signal Source",         width=14)
    tbl.add_column("Cost Model",            width=14)
    tbl.add_column("Trades", justify="right", width=8)
    tbl.add_column("Win Rate", justify="right", width=10)
    tbl.add_column("Profit Factor", justify="right", width=13)
    tbl.add_column("Gross P&L (₹)", justify="right", width=14)
    tbl.add_column("Total Taxes (₹)", justify="right", width=15)
    tbl.add_column("Net P&L (₹)", justify="right", width=14)
    tbl.add_column("Net Return (%)", justify="right", width=14)

    for r in [exp_a1, exp_a2, exp_b1, exp_b2]:
        p_col = "green" if r.total_net_pnl >= 0 else "red"
        p_sgn = "+" if r.total_net_pnl >= 0 else ""
        g_sgn = "+" if r.total_gross_pnl >= 0 else ""
        pf_col = "green" if r.profit_factor >= 1.5 else ("yellow" if r.profit_factor >= 1.0 else "red")

        tbl.add_row(
            f"[bold]{r.label}[/]",
            r.signal_source,
            r.cost_model,
            str(r.total_trades),
            f"{r.win_rate:.1f}%",
            f"[{pf_col}]{r.profit_factor:.2f}[/]",
            f"{g_sgn}₹{r.total_gross_pnl:>9,.2f}",
            f"[yellow]₹{r.total_costs:>9,.2f}[/]",
            f"[{p_col}]{p_sgn}₹{r.total_net_pnl:>9,.2f}[/]",
            f"[{p_col}]{p_sgn}{r.total_net_pnl_pct:>6.2f}%[/]",
        )

    console.print(tbl)
    console.print("\n[bold magenta]" + "═" * 120 + "[/]")
    console.print("[bold cyan]  FACTOR CONTRIBUTION BREAKDOWN:[/]")

    noise_impact = exp_b1.total_net_pnl - exp_a1.total_net_pnl
    tax_impact_a = exp_a2.total_net_pnl - exp_a1.total_net_pnl
    tax_impact_b = exp_b2.total_net_pnl - exp_b1.total_net_pnl

    console.print(f"  • [yellow]Indicator Noise Impact[/]  (Exp B vs Exp A Zero Cost) : "
                  f"[{'green' if noise_impact>=0 else 'red'}]{'+' if noise_impact>=0 else ''}₹{noise_impact:,.2f} "
                  f"({noise_impact/INITIAL_CAPITAL_INR*100:+.2f}%)  [dim]← Alpha destroyed by ETF price wiggles[/]")
    console.print(f"  • [yellow]CNC Tax Impact (Index)[/]  (Exp A Real vs Zero Cost)  : "
                  f"[red]₹{tax_impact_a:,.2f} ({tax_impact_a/INITIAL_CAPITAL_INR*100:+.2f}%)[/]  [dim]← Government STT/Exchange tax drag on Index signals[/]")
    console.print(f"  • [yellow]CNC Tax Impact (ETF)[/]    (Exp B Real vs Zero Cost)  : "
                  f"[red]₹{tax_impact_b:,.2f} ({tax_impact_b/INITIAL_CAPITAL_INR*100:+.2f}%)[/]  [dim]← Government STT/Exchange tax drag on noisy ETF signals[/]")
    console.print("[bold magenta]" + "═" * 120 + "[/]\n")


if __name__ == "__main__":
    main()
