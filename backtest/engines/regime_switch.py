# ─────────────────────────────────────────────────────────────────
#  backtest/engines/regime_switch.py
#  Walk-forward backtesting engine for VIX Regime-Switching Strategy.
#
#  Strategy: RegimeSwitchScorer (VIX-gated Mean-Rev vs Momentum)
#  Fees: STT + SEBI + brokerage modelled via india_costs.py
#  Sizing: risk-per-trade as % of capital / ATR-based SL distance
#
#  Usage:
#      from backtest.engines.regime_switch import RegimeSwitchBacktestEngine
#      bt = RegimeSwitchBacktestEngine("NIFTY50", "15m", bars=1400)
#      result = bt.run()
#      bt.print_report(result)
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime

import pandas as pd
from loguru import logger

from config.india_settings import (
    INITIAL_CAPITAL_INR,
    INDIA_MAX_RISK_PER_TRADE_PCT,
    INDIA_SIGNAL_THRESHOLD,
    INDIA_ATR_SL_MULT,
    INDIA_ATR_TP_MULT,
)
from data.india.nse_client import NSEClient
from indicators import add_all_indicators
from probability.regime_switch_scorer import RegimeSwitchScorer
from engine.india_costs import calculate_round_trip_cost


# ─────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────

@dataclass
class RegimeSwitchTrade:
    symbol:       str
    direction:    str
    entry_time:   datetime
    exit_time:    datetime
    entry_price:  float
    exit_price:   float
    qty:          float      # units
    pnl:          float      # realized P&L in INR
    pnl_pct:      float      # % return on capital risked
    exit_type:    str        # STOP_LOSS | TAKE_PROFIT | TIMEOUT
    probability:  float
    candles_held: int
    cost_inr:     float      # total transaction cost INR
    vix_at_entry: float      # VIX level when trade was entered
    regime_label: str        # MOMENTUM | MEAN_REV | HYBRID


@dataclass
class RegimeSwitchResult:
    symbol:          str
    interval:        str
    period_start:    datetime
    period_end:      datetime
    candles:         int
    initial_capital: float
    trades:          list[RegimeSwitchTrade] = field(default_factory=list)
    final_capital:   float = 0.0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> list[RegimeSwitchTrade]:
        return [t for t in self.trades if t.pnl > 0]

    @property
    def losing_trades(self) -> list[RegimeSwitchTrade]:
        return [t for t in self.trades if t.pnl <= 0]

    @property
    def win_rate(self) -> float:
        return len(self.winning_trades) / self.total_trades * 100 if self.trades else 0.0

    @property
    def total_pnl(self) -> float:
        return self.final_capital - self.initial_capital

    @property
    def total_pnl_pct(self) -> float:
        return self.total_pnl / self.initial_capital * 100

    @property
    def avg_win(self) -> float:
        wins = [t.pnl for t in self.winning_trades]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl for t in self.losing_trades]
        return sum(losses) / len(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gp = sum(t.pnl for t in self.winning_trades)
        gl = abs(sum(t.pnl for t in self.losing_trades))
        return round(gp / gl, 2) if gl else float("inf")

    @property
    def max_drawdown_pct(self) -> float:
        capital = self.initial_capital
        peak    = capital
        max_dd  = 0.0
        for t in self.trades:
            capital += t.pnl
            peak     = max(peak, capital)
            dd       = (peak - capital) / peak * 100
            max_dd   = max(max_dd, dd)
        return round(max_dd, 2)

    @property
    def total_costs_inr(self) -> float:
        return sum(t.cost_inr for t in self.trades)

    @property
    def avg_candles_held(self) -> float:
        return sum(t.candles_held for t in self.trades) / self.total_trades if self.trades else 0.0

    @property
    def best_trade(self) -> float:
        return max((t.pnl for t in self.trades), default=0.0)

    @property
    def worst_trade(self) -> float:
        return min((t.pnl for t in self.trades), default=0.0)

    @property
    def max_winning_streak(self) -> int:
        best = cur = 0
        for t in self.trades:
            cur = cur + 1 if t.pnl > 0 else 0
            best = max(best, cur)
        return best

    @property
    def max_losing_streak(self) -> int:
        best = cur = 0
        for t in self.trades:
            cur = cur + 1 if t.pnl <= 0 else 0
            best = max(best, cur)
        return best

    @property
    def monthly_returns(self) -> dict[str, float]:
        from collections import defaultdict
        capital = self.initial_capital
        month_start: dict[str, float] = {}
        monthly_pnl: dict[str, float] = defaultdict(float)
        for t in self.trades:
            mk = t.exit_time.strftime("%Y-%m")
            if mk not in month_start:
                month_start[mk] = capital
            monthly_pnl[mk] += t.pnl
            capital          += t.pnl
        return {mk: pnl / month_start[mk] * 100 for mk, pnl in monthly_pnl.items()}


# ─────────────────────────────────────────────────────────────────
#  Engine
# ─────────────────────────────────────────────────────────────────

class RegimeSwitchBacktestEngine:
    """
    Walk-forward backtesting engine for VIX Regime Switching.
    """

    def __init__(
        self,
        symbol:       str   = "NIFTY50",
        interval:     str   = "15m",
        bars:         int   = 1400,
        threshold:    float = INDIA_SIGNAL_THRESHOLD,
        atr_sl_mult:  float = INDIA_ATR_SL_MULT,
        atr_tp_mult:  float = INDIA_ATR_TP_MULT,
        capital:      float = INITIAL_CAPITAL_INR,
        vix:          float = 15.0,
        max_timeout:  int   = 0,
    ):
        self.symbol      = symbol
        self.interval    = interval
        self.bars        = bars
        self.threshold   = threshold
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.capital     = capital
        self.vix         = vix
        self.max_timeout = max_timeout
        self.client      = NSEClient()
        self.scorer      = RegimeSwitchScorer(
            symbol      = symbol,
            threshold   = threshold,
            current_vix = vix,
            interval    = interval,
        )

    # ── Data loading ──────────────────────────────────────────────

    def _load_data(self) -> pd.DataFrame:
        logger.info(f"Fetching {self.bars} × {self.interval} candles for {self.symbol}...")
        df = self.client.get_ohlcv(self.symbol, self.interval, self.bars)
        logger.info(f"Computing indicators on {len(df)} candles...")
        df = add_all_indicators(df)
        return df

    # ── Main run ──────────────────────────────────────────────────

    def run(self, external_df: pd.DataFrame | None = None) -> RegimeSwitchResult:
        df = external_df if external_df is not None else self._load_data()

        WARMUP = min(200, len(df) // 4)

        result = RegimeSwitchResult(
            symbol          = self.symbol,
            interval        = self.interval,
            period_start    = df.index[WARMUP],
            period_end      = df.index[-1],
            candles         = len(df) - WARMUP,
            initial_capital = self.capital,
            final_capital   = self.capital,
        )

        cash           = self.capital
        consec_losses  = 0
        cooldown_until = -1

        logger.info(f"Simulating {len(df) - WARMUP} candles (warmup={WARMUP})...")

        i = WARMUP
        while i < len(df) - 1:
            row      = df.iloc[i]
            df_slice = df.iloc[: i + 1]

            # Circuit breaker: 4 consecutive losses → 48-candle cooldown
            if i < cooldown_until:
                i += 1
                continue

            signal = self.scorer.score(row, df_slice)

            enter_long  = signal.direction == "LONG"  and signal.probability >= self.threshold
            enter_short = signal.direction == "SHORT" and signal.probability <= (100 - self.threshold)

            if not (enter_long or enter_short):
                i += 1
                continue

            # Recompute SL/TP from ATR
            atr = row.get("atr") if hasattr(row, "get") else row["atr"]
            if math.isnan(atr) or atr <= 0:
                atr = signal.entry_price * 0.008

            sl_dist = atr * self.atr_sl_mult
            entry   = signal.entry_price

            if signal.direction == "LONG":
                sl  = round(entry - sl_dist, 2)
                tp  = round(entry + sl_dist * self.atr_tp_mult, 2)
            else:
                sl  = round(entry + sl_dist, 2)
                tp  = round(entry - sl_dist * self.atr_tp_mult, 2)

            signal = replace(signal, stop_loss=sl, take_profit1=tp, take_profit2=tp, risk_amount=sl_dist)

            # Size: risk INDIA_MAX_RISK_PER_TRADE_PCT% of cash / SL distance
            risk_inr = cash * INDIA_MAX_RISK_PER_TRADE_PCT / 100
            qty      = risk_inr / sl_dist if sl_dist > 0 else 0
            if qty <= 0:
                i += 1
                continue

            trade = self._simulate_forward(df, signal, i, qty, entry)

            if trade:
                cash += trade.pnl
                result.trades.append(trade)

                if trade.exit_type == "STOP_LOSS":
                    consec_losses += 1
                    if consec_losses >= 4:
                        cooldown_until = i + 48
                        consec_losses  = 0
                        logger.warning(f"Circuit breaker: 4 losses — cooling down 48 candles from {i}")
                else:
                    consec_losses = 0

                i += trade.candles_held
                continue

            i += 1

        result.final_capital = round(cash, 2)
        logger.success(f"Regime-Switch backtest done: {result.total_trades} trades, "
                       f"P&L ₹{result.total_pnl:+,.2f} ({result.total_pnl_pct:+.2f}%)")
        return result

    # ── Forward simulation ────────────────────────────────────────

    def _simulate_forward(
        self,
        df:        pd.DataFrame,
        signal,
        entry_idx: int,
        qty:       float,
        entry:     float,
    ) -> RegimeSwitchTrade | None:
        direction = signal.direction
        sl        = signal.stop_loss
        tp        = signal.take_profit1
        timeout   = self.max_timeout if self.max_timeout > 0 else len(df)

        for j in range(entry_idx + 1, min(entry_idx + timeout + 1, len(df))):
            candle       = df.iloc[j]
            high, low    = candle["high"], candle["low"]
            candles_held = j - entry_idx

            if direction == "LONG":
                sl_hit = low  <= sl
                tp_hit = high >= tp
            else:
                sl_hit = high >= sl
                tp_hit = low  <= tp

            if sl_hit or tp_hit:
                exit_price = sl if sl_hit else tp
                exit_type  = "STOP_LOSS" if sl_hit else "TAKE_PROFIT"
                return self._make_trade(signal, df, entry_idx, j, entry, exit_price, exit_type, qty, candles_held)

        if self.max_timeout > 0:
            j            = min(entry_idx + timeout, len(df) - 1)
            exit_price   = df.iloc[j]["close"]
            candles_held = j - entry_idx
            return self._make_trade(signal, df, entry_idx, j, entry, exit_price, "TIMEOUT", qty, candles_held)

        return None

    def _make_trade(
        self,
        signal,
        df:           pd.DataFrame,
        entry_idx:    int,
        exit_idx:     int,
        entry_price:  float,
        exit_price:   float,
        exit_type:    str,
        qty:          float,
        candles_held: int,
    ) -> RegimeSwitchTrade:
        if signal.direction == "LONG":
            gross_pnl = (exit_price - entry_price) * qty
        else:
            gross_pnl = (entry_price - exit_price) * qty

        buy_cost, sell_cost = calculate_round_trip_cost(entry_price, exit_price, qty, "FO_FUTURES")
        cost_inr     = buy_cost.total + sell_cost.total
        net_pnl      = gross_pnl - cost_inr
        pnl_pct      = net_pnl / (entry_price * qty) * 100

        # Extract regime label from signal reason string
        reason_str = signal.reason or ""
        if "MOMENTUM REGIME" in reason_str:
            reg_label = "MOMENTUM"
        elif "MEAN_REV REGIME" in reason_str:
            reg_label = "MEAN_REV"
        else:
            reg_label = "HYBRID"

        return RegimeSwitchTrade(
            symbol       = self.symbol,
            direction    = signal.direction,
            entry_time   = df.index[entry_idx].to_pydatetime(),
            exit_time    = df.index[exit_idx].to_pydatetime(),
            entry_price  = entry_price,
            exit_price   = round(exit_price, 2),
            qty          = qty,
            pnl          = round(net_pnl, 2),
            pnl_pct      = round(pnl_pct, 4),
            exit_type    = exit_type,
            probability  = signal.probability,
            candles_held = candles_held,
            cost_inr     = round(cost_inr, 2),
            vix_at_entry = round(self.scorer.current_vix, 2),
            regime_label = reg_label,
        )

    # ── Report ────────────────────────────────────────────────────

    def print_report(self, r: RegimeSwitchResult) -> None:
        from rich.console import Console
        from rich.table   import Table
        from rich         import box
        from collections  import Counter

        c = Console()

        def sep():
            c.print("[bold blue]" + "═" * 76 + "[/]")

        sign    = "+" if r.total_pnl >= 0 else ""
        pnl_col = "green" if r.total_pnl >= 0 else "red"

        sep()
        c.print(f"[bold cyan]  VIX REGIME-SWITCHING BACKTEST — {r.symbol}  {r.interval}[/]")
        c.print("[dim]  Strategy: VIX<18 Momentum | VIX 18-25 Hybrid Blend | VIX>25 Mean-Reversion[/]")
        sep()
        c.print(f"  [dim]Period   :[/]  {r.period_start.strftime('%Y-%m-%d')}  →  {r.period_end.strftime('%Y-%m-%d')}")
        c.print(f"  [dim]Candles  :[/]  {r.candles}  (warmup excluded)")
        c.print(f"  [dim]Threshold:[/]  LONG/SHORT ≥ {self.threshold}%  |  SL={self.atr_sl_mult}×ATR  TP={self.atr_tp_mult}R")
        c.print(f"  [dim]VIX (start):[/]  {self.vix:.1f}")

        c.print(f"\n  [bold]── Performance ─────────────────────────────────────[/]")
        c.print(f"  Initial Capital  : [cyan]₹{r.initial_capital:>12,.2f}[/]")
        c.print(f"  Final Capital    : [cyan]₹{r.final_capital:>12,.2f}[/]")
        c.print(f"  Total P&L        : [{pnl_col}]{sign}₹{r.total_pnl:>10,.2f}  ({sign}{r.total_pnl_pct:.2f}%)[/]")
        c.print(f"  Max Drawdown     : [red]{r.max_drawdown_pct:.2f}%[/]")
        c.print(f"  Total Costs      : [yellow]₹{r.total_costs_inr:,.2f}[/]")

        c.print(f"\n  [bold]── Trade Statistics ─────────────────────────────────[/]")
        c.print(f"  Total Trades     : [bold]{r.total_trades}[/]")
        c.print(f"  Win Rate         : [yellow]{r.win_rate:.1f}%[/]  ({len(r.winning_trades)}W / {len(r.losing_trades)}L)")
        c.print(f"  Profit Factor    : [{'green' if r.profit_factor >= 1.5 else 'yellow'}]{r.profit_factor:.2f}[/]  [dim](≥1.5 good)[/]")
        c.print(f"  Avg Win          : [green]+₹{r.avg_win:,.2f}[/]")
        c.print(f"  Avg Loss         : [red]₹{r.avg_loss:,.2f}[/]")
        c.print(f"  Best Trade       : [green]+₹{r.best_trade:,.2f}[/]")
        c.print(f"  Worst Trade      : [red]₹{r.worst_trade:,.2f}[/]")
        c.print(f"  Max Win Streak   : [green]{r.max_winning_streak}[/]")
        c.print(f"  Max Loss Streak  : [red]{r.max_losing_streak}[/]")
        c.print(f"  Avg Candles Held : {r.avg_candles_held:.1f}")

        # Breakdown by Regime
        regimes_count = Counter(t.regime_label for t in r.trades)
        c.print(f"\n  [bold]── Regime Breakdown ─────────────────────────────────[/]")
        for reg in ["MOMENTUM", "HYBRID", "MEAN_REV"]:
            reg_trades = [t for t in r.trades if t.regime_label == reg]
            cnt = len(reg_trades)
            if cnt > 0:
                reg_pnl = sum(t.pnl for t in reg_trades)
                reg_wr  = len([t for t in reg_trades if t.pnl > 0]) / cnt * 100
                r_col   = "green" if reg_pnl >= 0 else "red"
                r_sign  = "+" if reg_pnl >= 0 else ""
                c.print(f"  [cyan]{reg:<12}[/]  {cnt:>3} trades  |  WinRate: [yellow]{reg_wr:>4.1f}%[/]  |  P&L: [{r_col}]{r_sign}₹{reg_pnl:>9,.2f}[/]")
            else:
                c.print(f"  [dim]{reg:<12}    0 trades[/]")

        exits = Counter(t.exit_type for t in r.trades)
        c.print(f"\n  [bold]── Exit Distribution ────────────────────────────────[/]")
        for etype, count in exits.most_common():
            pct     = count / r.total_trades * 100 if r.total_trades else 0
            col     = "green" if etype == "TAKE_PROFIT" else ("yellow" if etype == "TIMEOUT" else "red")
            c.print(f"  [{col}]{etype:<20}[/]  {count:>3}  ({pct:.1f}%)")

        c.print(f"\n  [bold]── Monthly Returns ──────────────────────────────────[/]")
        returns = r.monthly_returns
        if returns:
            for month, pct in sorted(returns.items()):
                m_col  = "green" if pct >= 0 else "red"
                m_sign = "+" if pct >= 0 else ""
                c.print(f"  {month} : [{m_col}]{m_sign}{pct:.2f}%[/]")
        else:
            c.print("  [dim]Not enough data[/]")

        c.print(f"\n  [bold]── Trade Log (last 20) ──────────────────────────────[/]")
        tbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
        tbl.add_column("Entry",      width=14)
        tbl.add_column("Dir",        width=5)
        tbl.add_column("Regime",     width=9)
        tbl.add_column("Entry ₹",    justify="right", width=9)
        tbl.add_column("Exit ₹",     justify="right", width=9)
        tbl.add_column("Exit Type",  width=12)
        tbl.add_column("P&L",        justify="right", width=11)
        tbl.add_column("Prob",       justify="right", width=6)
        tbl.add_column("VIX",        justify="right", width=5)

        for trade in r.trades[-20:]:
            pc  = "green" if trade.pnl >= 0 else "red"
            sgn = "+" if trade.pnl >= 0 else ""
            dc  = "green" if trade.direction == "LONG" else "red"
            reg_c = "cyan" if trade.regime_label == "MOMENTUM" else ("yellow" if trade.regime_label == "HYBRID" else "magenta")
            tbl.add_row(
                trade.entry_time.strftime("%m-%d %H:%M"),
                f"[{dc}]{trade.direction[:4]}[/]",
                f"[{reg_c}]{trade.regime_label[:8]}[/]",
                f"₹{trade.entry_price:>7,.1f}",
                f"₹{trade.exit_price:>7,.1f}",
                trade.exit_type[:11],
                f"[{pc}]{sgn}₹{trade.pnl:>7,.2f}[/]",
                f"{trade.probability:.1f}%",
                f"{trade.vix_at_entry:.1f}",
            )
        c.print(tbl)
        sep()
