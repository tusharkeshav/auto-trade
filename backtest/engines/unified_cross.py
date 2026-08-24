# ─────────────────────────────────────────────────────────────────
#  backtest/engines/unified_cross.py
#  Walk-forward backtesting engine for "Momentum Crosses Mean" Strategy.
#
#  Features:
#    - UnifiedCrossScorer (3 Hardened Shields)
#    - Asymmetric 1:2.5 Risk-Reward Targets (SL = 1.0× ATR, TP = 2.5× ATR)
#    - Dynamic ATR Trailing Stop-Loss:
#        * At +1.0R profit: Trails SL to Break-Even (entry_price)
#        * At +1.5R profit: Trails SL to +0.5R profit lock-in
#    - Accurate F&O Index Futures transaction costs (FO_FUTURES)
#    - Circuit breaker: 4 consecutive losses → 48-candle cooldown
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
)
from data.india.nse_client import NSEClient
from indicators import add_all_indicators
from probability.unified_cross_scorer import UnifiedCrossScorer
from engine.india_costs import calculate_round_trip_cost


@dataclass
class UnifiedCrossTrade:
    symbol:       str
    direction:    str
    entry_time:   datetime
    exit_time:    datetime
    entry_price:  float
    exit_price:   float
    qty:          float
    pnl:          float
    pnl_pct:      float
    exit_type:    str        # STOP_LOSS | TAKE_PROFIT | TRAIL_STOP | TIMEOUT
    probability:  float
    candles_held: int
    cost_inr:     float
    vix_at_entry: float
    setup_label:  str        # PULLBACK | BAND_BOUNCE | DUAL_CONFIRM


@dataclass
class UnifiedCrossResult:
    symbol:          str
    interval:        str
    period_start:    datetime
    period_end:      datetime
    candles:         int
    initial_capital: float
    trades:          list[UnifiedCrossTrade] = field(default_factory=list)
    final_capital:   float = 0.0

    @property
    def total_trades(self) -> int: return len(self.trades)

    @property
    def winning_trades(self) -> list[UnifiedCrossTrade]: return [t for t in self.trades if t.pnl > 0]

    @property
    def losing_trades(self) -> list[UnifiedCrossTrade]: return [t for t in self.trades if t.pnl <= 0]

    @property
    def win_rate(self) -> float:
        return len(self.winning_trades) / self.total_trades * 100 if self.trades else 0.0

    @property
    def total_pnl(self) -> float: return self.final_capital - self.initial_capital

    @property
    def total_pnl_pct(self) -> float: return self.total_pnl / self.initial_capital * 100

    @property
    def avg_win(self) -> float:
        w = [t.pnl for t in self.winning_trades]
        return sum(w) / len(w) if w else 0.0

    @property
    def avg_loss(self) -> float:
        l = [t.pnl for t in self.losing_trades]
        return sum(l) / len(l) if l else 0.0

    @property
    def profit_factor(self) -> float:
        gp = sum(t.pnl for t in self.winning_trades)
        gl = abs(sum(t.pnl for t in self.losing_trades))
        return round(gp / gl, 2) if gl else float("inf")

    @property
    def max_drawdown_pct(self) -> float:
        capital = self.initial_capital
        peak, max_dd = capital, 0.0
        for t in self.trades:
            capital += t.pnl
            peak     = max(peak, capital)
            dd       = (peak - capital) / peak * 100
            max_dd   = max(max_dd, dd)
        return round(max_dd, 2)

    @property
    def total_costs_inr(self) -> float: return sum(t.cost_inr for t in self.trades)

    @property
    def avg_candles_held(self) -> float:
        return sum(t.candles_held for t in self.trades) / self.total_trades if self.trades else 0.0

    @property
    def best_trade(self) -> float: return max((t.pnl for t in self.trades), default=0.0)

    @property
    def worst_trade(self) -> float: return min((t.pnl for t in self.trades), default=0.0)

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
        capital, month_start, monthly_pnl = self.initial_capital, {}, defaultdict(float)
        for t in self.trades:
            mk = t.exit_time.strftime("%Y-%m")
            if mk not in month_start: month_start[mk] = capital
            monthly_pnl[mk] += t.pnl
            capital          += t.pnl
        return {mk: pnl / month_start[mk] * 100 for mk, pnl in monthly_pnl.items()}


class UnifiedCrossBacktestEngine:
    """
    Walk-forward backtesting engine for Unified Cross Strategy.
    """

    def __init__(
        self,
        symbol:       str   = "NIFTY50",
        interval:     str   = "15m",
        bars:         int   = 1400,
        threshold:    float = INDIA_SIGNAL_THRESHOLD,
        atr_sl_mult:  float = 1.0,
        atr_tp_mult:  float = 2.5,
        capital:      float = INITIAL_CAPITAL_INR,
        vix:          float = 15.0,
        max_timeout:  int   = 0,
        trade_type:   str | None = None,
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
        self.trade_type  = trade_type if trade_type else ("CNC" if symbol.endswith(".NS") else "FO_FUTURES")
        self.client      = NSEClient()
        self.scorer      = UnifiedCrossScorer(
            symbol      = symbol,
            threshold   = threshold,
            atr_sl_mult = atr_sl_mult,
            atr_tp_mult = atr_tp_mult,
            interval    = interval,
            current_vix = vix,
        )

    def _load_data(self) -> pd.DataFrame:
        logger.info(f"Fetching {self.bars} × {self.interval} candles for {self.symbol}...")
        df = self.client.get_ohlcv(self.symbol, self.interval, self.bars)
        logger.info(f"Computing indicators on {len(df)} candles...")
        df = add_all_indicators(df)
        return df

    def run(self, external_df: pd.DataFrame | None = None) -> UnifiedCrossResult:
        df = external_df if external_df is not None else self._load_data()
        WARMUP = min(200, len(df) // 4)

        result = UnifiedCrossResult(
            symbol          = self.symbol,
            interval        = self.interval,
            period_start    = df.index[WARMUP],
            period_end      = df.index[-1],
            candles         = len(df) - WARMUP,
            initial_capital = self.capital,
            final_capital   = self.capital,
        )
        cash = self.capital
        consec_losses = 0
        cooldown_until = -1

        logger.info(f"Simulating {len(df) - WARMUP} candles (warmup={WARMUP})...")

        i = WARMUP
        while i < len(df) - 1:
            row = df.iloc[i]
            df_slice = df.iloc[: i + 1]

            if i < cooldown_until:
                i += 1
                continue

            signal = self.scorer.score(row, df_slice)
            if not signal.is_tradeable() or (self.trade_type == "CNC" and signal.direction == "SHORT"):
                i += 1
                continue

            atr = row.get("atr") if hasattr(row, "get") else row["atr"]
            if math.isnan(atr) or atr <= 0:
                atr = signal.entry_price * 0.006

            sl_dist = atr * self.atr_sl_mult
            entry   = signal.entry_price

            if signal.direction == "LONG":
                sl = round(entry - sl_dist, 2)
                tp = round(entry + sl_dist * self.atr_tp_mult, 2)
            else:
                sl = round(entry + sl_dist, 2)
                tp = round(entry - sl_dist * self.atr_tp_mult, 2)

            signal = replace(signal, stop_loss=sl, take_profit1=tp, take_profit2=tp, risk_amount=sl_dist)

            risk_inr = cash * INDIA_MAX_RISK_PER_TRADE_PCT / 100
            qty      = risk_inr / sl_dist if sl_dist > 0 else 0
            if self.trade_type == "CNC" and entry > 0:
                qty = min(qty, cash / entry)
            if qty <= 0:
                i += 1
                continue

            trade = self._simulate_forward(df, signal, i, qty, entry, sl_dist)

            if trade:
                cash += trade.pnl
                result.trades.append(trade)

                if trade.exit_type in ("STOP_LOSS", "TRAIL_STOP") and trade.pnl <= 0:
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
        logger.success(f"Unified Cross backtest done: {result.total_trades} trades, "
                       f"P&L ₹{result.total_pnl:+,.2f} ({result.total_pnl_pct:+.2f}%)")
        return result

    def _simulate_forward(
        self,
        df:        pd.DataFrame,
        signal,
        entry_idx: int,
        qty:       float,
        entry:     float,
        sl_dist:   float,
    ) -> UnifiedCrossTrade | None:
        direction = signal.direction
        sl        = signal.stop_loss
        tp        = signal.take_profit1
        timeout   = self.max_timeout if self.max_timeout > 0 else len(df)

        for j in range(entry_idx + 1, min(entry_idx + timeout + 1, len(df))):
            candle       = df.iloc[j]
            high, low    = candle["high"], candle["low"]
            candles_held = j - entry_idx

            # ── Dynamic ATR Trailing Stop-Loss ────────────────────────
            if direction == "LONG":
                # Check target hit or stop hit
                if low <= sl:
                    exit_type = "TRAIL_STOP" if sl >= entry else "STOP_LOSS"
                    return self._make_trade(signal, df, entry_idx, j, entry, sl, exit_type, qty, candles_held)
                if high >= tp:
                    return self._make_trade(signal, df, entry_idx, j, entry, tp, "TAKE_PROFIT", qty, candles_held)

                # Trail SL upward as price moves in our favor
                if high >= entry + sl_dist * 1.5 and sl < entry + sl_dist * 0.5:
                    sl = round(entry + sl_dist * 0.5, 2)   # Lock in +0.5R profit
                elif high >= entry + sl_dist * 1.0 and sl < entry:
                    sl = round(entry, 2)                   # Trail to Break-Even
            else:
                if high >= sl:
                    exit_type = "TRAIL_STOP" if sl <= entry else "STOP_LOSS"
                    return self._make_trade(signal, df, entry_idx, j, entry, sl, exit_type, qty, candles_held)
                if low <= tp:
                    return self._make_trade(signal, df, entry_idx, j, entry, tp, "TAKE_PROFIT", qty, candles_held)

                # Trail SL downward for short trades
                if low <= entry - sl_dist * 1.5 and sl > entry - sl_dist * 0.5:
                    sl = round(entry - sl_dist * 0.5, 2)   # Lock in +0.5R profit
                elif low <= entry - sl_dist * 1.0 and sl > entry:
                    sl = round(entry, 2)                   # Trail to Break-Even

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
    ) -> UnifiedCrossTrade:
        if signal.direction == "LONG":
            gross_pnl = (exit_price - entry_price) * qty
        else:
            gross_pnl = (entry_price - exit_price) * qty

        buy_cost, sell_cost = calculate_round_trip_cost(entry_price, exit_price, qty, self.trade_type)
        cost_inr     = buy_cost.total + sell_cost.total
        net_pnl      = gross_pnl - cost_inr
        pnl_pct      = net_pnl / (entry_price * qty) * 100

        reason_str = signal.reason or ""
        if "DUAL CONFIRM" in reason_str:
            setup_lbl = "DUAL_CONFIRM"
        elif "BAND BOUNCE" in reason_str:
            setup_lbl = "BAND_BOUNCE"
        else:
            setup_lbl = "PULLBACK"

        return UnifiedCrossTrade(
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
            setup_label  = setup_lbl,
        )

    def print_report(self, r: UnifiedCrossResult) -> None:
        from rich.console import Console
        from rich.table   import Table
        from rich         import box
        from collections  import Counter

        c = Console()
        def sep(): c.print("[bold blue]" + "═" * 78 + "[/]")

        sign    = "+" if r.total_pnl >= 0 else ""
        pnl_col = "green" if r.total_pnl >= 0 else "red"

        sep()
        c.print(f"[bold cyan]  UNIFIED CROSS BACKTEST — {r.symbol}  {r.interval}[/]")
        c.print("[dim]  Strategy: VWAP/EMA Pullback Cross + BB Reversal Cross (1:2.5 Asymmetric R)[/]")
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

        # Breakdown by Setup
        c.print(f"\n  [bold]── Setup Breakdown ───────────────────────────────────[/]")
        for setup in ["PULLBACK", "BAND_BOUNCE", "DUAL_CONFIRM"]:
            st_trades = [t for t in r.trades if t.setup_label == setup]
            cnt = len(st_trades)
            if cnt > 0:
                st_pnl = sum(t.pnl for t in st_trades)
                st_wr  = len([t for t in st_trades if t.pnl > 0]) / cnt * 100
                r_col  = "green" if st_pnl >= 0 else "red"
                r_sign = "+" if st_pnl >= 0 else ""
                c.print(f"  [cyan]{setup:<14}[/]  {cnt:>3} trades  |  WinRate: [yellow]{st_wr:>4.1f}%[/]  |  P&L: [{r_col}]{r_sign}₹{st_pnl:>9,.2f}[/]")
            else:
                c.print(f"  [dim]{setup:<14}    0 trades[/]")

        exits = Counter(t.exit_type for t in r.trades)
        c.print(f"\n  [bold]── Exit Distribution ────────────────────────────────[/]")
        for etype, count in exits.most_common():
            pct = count / r.total_trades * 100 if r.total_trades else 0
            col = "green" if etype in ("TAKE_PROFIT", "TRAIL_STOP") else ("yellow" if etype == "TIMEOUT" else "red")
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
        tbl.add_column("Setup",      width=12)
        tbl.add_column("Entry ₹",    justify="right", width=9)
        tbl.add_column("Exit ₹",     justify="right", width=9)
        tbl.add_column("Exit Type",  width=11)
        tbl.add_column("P&L",        justify="right", width=11)
        tbl.add_column("Prob",       justify="right", width=6)
        tbl.add_column("VIX",        justify="right", width=5)

        for trade in r.trades[-20:]:
            pc  = "green" if trade.pnl >= 0 else "red"
            sgn = "+" if trade.pnl >= 0 else ""
            dc  = "green" if trade.direction == "LONG" else "red"
            st_c = "cyan" if trade.setup_label == "PULLBACK" else ("yellow" if trade.setup_label == "BAND_BOUNCE" else "magenta")
            tbl.add_row(
                trade.entry_time.strftime("%m-%d %H:%M"),
                f"[{dc}]{trade.direction[:4]}[/]",
                f"[{st_c}]{trade.setup_label[:11]}[/]",
                f"₹{trade.entry_price:>7,.1f}",
                f"₹{trade.exit_price:>7,.1f}",
                trade.exit_type[:10],
                f"[{pc}]{sgn}₹{trade.pnl:>7,.2f}[/]",
                f"{trade.probability:.1f}%",
                f"{trade.vix_at_entry:.1f}",
            )
        c.print(tbl)
        sep()
