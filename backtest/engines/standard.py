# ─────────────────────────────────────────────────────────────────
#  backtest/engine.py
#
#  Simulates the trading strategy on historical OHLCV data.
#
#  Methodology:
#    - Walk forward candle-by-candle (no lookahead bias)
#    - Enter when probability ≥ threshold on candle CLOSE
#    - Find SL/TP hits using HIGH/LOW of subsequent candles
#    - Conflict rule: if SL and TP hit in same candle → SL wins
#      (conservative — avoids optimistic assumptions)
#    - Partial TP1 at 50%, stop moves to breakeven
#    - TP2 closes remaining 50%
#
#  Usage:
#      from backtest import BacktestEngine
#      bt = BacktestEngine("BTCUSDT", "1h", candles=500)
#      result = bt.run()
#      bt.print_report(result)
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations
from dataclasses import replace

import math
from dataclasses import dataclass, field
from datetime    import datetime

import pandas as pd
from loguru  import logger

from config.settings           import INITIAL_CAPITAL_USDT, MAX_RISK_PER_TRADE_PCT
from data.binance_client        import BinanceClient
from data.cache                 import DataCache
from indicators                 import add_all_indicators
from probability.signal_scorer  import SignalScorer, TradeSignal


# ── Trading fees (Binance spot taker: 0.1% per side = 0.2% round-trip) ────────
TAKER_FEE_RATE = 0.001  # 0.1% per trade leg


# ─────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    """Record of one simulated trade during backtest."""
    symbol:      str
    direction:   str
    entry_time:  datetime
    exit_time:   datetime
    entry_price: float
    exit_price:  float
    size:        float
    pnl:         float        # realized P&L in USDT
    pnl_pct:     float        # % return on capital used
    exit_type:   str          # STOP_LOSS | TAKE_PROFIT_1 | TAKE_PROFIT_2 | BREAKEVEN
    probability: float
    candles_held: int


@dataclass
class BacktestResult:
    symbol:       str
    interval:     str
    period_start: datetime
    period_end:   datetime
    candles:      int
    initial_capital: float

    trades:       list[BacktestTrade] = field(default_factory=list)
    final_capital: float = 0.0

    # ── Computed metrics ──────────────────────────────────────────

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> list[BacktestTrade]:
        return [t for t in self.trades if t.pnl > 0]

    @property
    def losing_trades(self) -> list[BacktestTrade]:
        return [t for t in self.trades if t.pnl <= 0]

    @property
    def win_rate(self) -> float:
        if not self.trades: return 0.0
        return len(self.winning_trades) / self.total_trades * 100

    @property
    def total_pnl(self) -> float:
        return self.final_capital - self.initial_capital

    @property
    def total_pnl_pct(self) -> float:
        return (self.total_pnl / self.initial_capital) * 100

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
        gross_profit = sum(t.pnl for t in self.winning_trades)
        gross_loss   = abs(sum(t.pnl for t in self.losing_trades))
        return round(gross_profit / gross_loss, 2) if gross_loss else float("inf")

    @property
    def max_drawdown_pct(self) -> float:
        """Maximum peak-to-trough drawdown on the running capital curve."""
        capital   = self.initial_capital
        peak      = capital
        max_dd    = 0.0
        for t in self.trades:
            capital += t.pnl
            peak     = max(peak, capital)
            dd       = (peak - capital) / peak * 100
            max_dd   = max(max_dd, dd)
        return round(max_dd, 2)

    @property
    def avg_candles_held(self) -> float:
        if not self.trades: return 0.0
        return sum(t.candles_held for t in self.trades) / self.total_trades

    @property
    def best_trade(self) -> float:
        return max((t.pnl for t in self.trades), default=0.0)

    @property
    def worst_trade(self) -> float:
        return min((t.pnl for t in self.trades), default=0.0)

    @property
    def max_winning_streak(self) -> int:
        max_streak, current_streak = 0, 0
        for t in self.trades:
            if t.pnl > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    @property
    def max_losing_streak(self) -> int:
        max_streak, current_streak = 0, 0
        for t in self.trades:
            if t.pnl <= 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    @property
    def monthly_returns(self) -> dict[str, float]:
        from collections import defaultdict
        capital = self.initial_capital
        month_start_capital = {}
        monthly_pnl = defaultdict(float)

        for t in self.trades:
            month_key = t.exit_time.strftime("%Y-%m")
            if month_key not in month_start_capital:
                month_start_capital[month_key] = capital
            
            monthly_pnl[month_key] += t.pnl
            capital += t.pnl
            
        returns = {}
        for mk, pnl in monthly_pnl.items():
            returns[mk] = (pnl / month_start_capital[mk]) * 100
        return returns


# ─────────────────────────────────────────────────────────────────
#  Core engine
# ─────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Walk-forward backtesting engine.

    Parameters:
        symbol    : Trading pair (e.g. "BTCUSDT")
        interval  : Candle timeframe (e.g. "1h")
        candles   : How many historical candles to test on (max 1000)
        threshold : Minimum probability to enter a trade
        capital   : Starting paper capital in USDT
    """

    def __init__(
        self,
        symbol:          str   = "BTCUSDT",
        interval:        str   = "1h",
        candles:         int   = 500,
        threshold:       float = 70.0,    # LONG entry: probability >= threshold
        short_threshold: float = 35.0,    # SHORT entry: probability <= short_threshold
        atr_sl_mult:     float = 1.0,     # ATR multiplier for stop-loss distance
        capital:         float = INITIAL_CAPITAL_USDT,
    ):
        self.symbol          = symbol
        self.interval        = interval
        self.candles         = min(candles, 1000)
        self.threshold       = threshold
        self.short_threshold = short_threshold
        self.atr_sl_mult     = atr_sl_mult
        self.capital         = capital
        self.client          = BinanceClient()
        self.cache           = DataCache()
        self.scorer          = SignalScorer(symbol=symbol, long_threshold=threshold, short_threshold=short_threshold)

    # ─────────────────────────────────────────────────────────────
    #  Run
    # ─────────────────────────────────────────────────────────────

    def _load_data(self) -> pd.DataFrame:
        """Load from local cache if available, otherwise fetch from Binance and cache."""
        df = self.cache.load(self.symbol, self.interval)
        if df is not None:
            return df

        # Cache miss — fetch live and save for next time
        logger.info(f"Cache miss. Fetching {self.candles} {self.interval} candles for {self.symbol}...")
        df = self.client.get_ohlcv(self.symbol, self.interval, self.candles)
        logger.info("Computing indicators...")
        df = add_all_indicators(df)
        self.cache.save(df, self.symbol, self.interval)
        return df

    def run(self, external_df: pd.DataFrame = None) -> BacktestResult:
        if external_df is not None:
            df = external_df
        else:
            df = self._load_data()

        # Warm-up: need at least 200 candles for all indicators to stabilise
        # If external_df is passed, we assume it's already warmed up, but we'll keep the logic safe.
        WARMUP = 200 if len(df) > 400 else 0
        
        result = BacktestResult(
            symbol          = self.symbol,
            interval        = self.interval,
            period_start    = df.index[WARMUP],
            period_end      = df.index[-1],
            candles         = len(df) - WARMUP,
            initial_capital = self.capital,
            final_capital   = self.capital,
        )

        cash          = self.capital
        in_position   = False
        signal_count  = 0
        consec_losses = 0          # Step 3: consecutive-loss counter
        cooldown_until = -1        # Step 3: candle index where cooldown ends

        logger.info(f"Simulating trades from candle {WARMUP} to {len(df) - 1}...")

        i = WARMUP
        while i < len(df) - 1:   # -1: need at least one forward candle
            row = df.iloc[i]

            if not in_position:
                # Step 3: Circuit breaker — skip if we're in cooldown after 4 losses
                if i < cooldown_until:
                    i += 1
                    continue

                # Score this candle
                signal = self.scorer.score(row)

                if (signal.probability >= self.threshold      and signal.direction == "LONG") or \
                   (signal.probability <= self.short_threshold and signal.direction == "SHORT"):
                    signal_count += 1

                    # ── Override SL/TP with custom ATR multiplier ─────────────
                    atr      = row["atr"] if not math.isnan(row["atr"]) else signal.entry_price * 0.004
                    sl_dist  = atr * self.atr_sl_mult
                    entry    = signal.entry_price

                    if signal.direction == "LONG":
                        new_sl  = round(entry - sl_dist, 2)
                        new_tp1 = round(entry + sl_dist * 1.5,  2)   # 1.5R (All-in, All-out)
                        new_tp2 = new_tp1
                    else:  # SHORT
                        new_sl  = round(entry + sl_dist, 2)
                        new_tp1 = round(entry - sl_dist * 1.5,  2)   # 1.5R (All-in, All-out)
                        new_tp2 = new_tp1

                    signal = replace(
                        signal,
                        stop_loss    = new_sl,
                        take_profit1 = new_tp1,
                        take_profit2 = new_tp2,
                        risk_amount  = round(sl_dist, 2),
                    )

                    # Position sizing: risk MAX_RISK_PER_TRADE_PCT% of cash
                    risk_budget   = cash * (MAX_RISK_PER_TRADE_PCT / 100)
                    risk_per_unit = sl_dist

                    if risk_per_unit == 0:
                        i += 1
                        continue

                    size = risk_budget / risk_per_unit

                    # Simulate forward to find SL/TP hit
                    trade = self._simulate_forward(
                        df       = df,
                        signal   = signal,
                        entry_idx= i,
                        size     = size,
                    )

                    if trade:
                        cash          += trade.pnl
                        result.trades.append(trade)
                        in_position    = False

                        # Step 3: track consecutive losses → trigger cooldown
                        if trade.exit_type == "STOP_LOSS":
                            consec_losses += 1
                            if consec_losses >= 4:
                                cooldown_until = i + 48   # 48 × 15m = 12 hours pause
                                consec_losses = 0
                        else:
                            consec_losses = 0   # any win/BE resets the counter

                        i += trade.candles_held
                        continue

            i += 1

        result.final_capital = round(cash, 4)
        logger.success(f"Backtest complete: {result.total_trades} trades simulated.")
        return result

    # ─────────────────────────────────────────────────────────────
    #  Forward simulation
    # ─────────────────────────────────────────────────────────────

    def _simulate_forward(
        self,
        df:        pd.DataFrame,
        signal:    TradeSignal,
        entry_idx: int,
        size:      float,
    ) -> BacktestTrade | None:
        """
        Walk forward from entry_idx scanning each candle's HIGH and LOW
        to find the first SL or TP hit.

        All-In, All-Out rule:
          100% position exits at TP (1.5R) or SL (1.0R). No breakeven.
          Conservative conflict rule: If both hit in same candle, SL wins.
        """
        direction = signal.direction
        entry     = signal.entry_price
        sl        = signal.stop_loss
        tp        = signal.take_profit1  # Using tp1 as the single target

        for j in range(entry_idx + 1, len(df)):
            candle      = df.iloc[j]
            candle_high = candle["high"]
            candle_low  = candle["low"]
            candles_held = j - entry_idx

            if direction == "LONG":
                sl_hit = candle_low <= sl
                tp_hit = candle_high >= tp

                if sl_hit:
                    return self._make_trade(signal, df, entry_idx, j, sl, "STOP_LOSS", size, 0.0, candles_held)
                if tp_hit:
                    return self._make_trade(signal, df, entry_idx, j, tp, "TAKE_PROFIT", size, 0.0, candles_held)

            else:  # SHORT
                sl_hit = candle_high >= sl
                tp_hit = candle_low <= tp

                if sl_hit:
                    return self._make_trade(signal, df, entry_idx, j, sl, "STOP_LOSS", size, 0.0, candles_held)
                if tp_hit:
                    return self._make_trade(signal, df, entry_idx, j, tp, "TAKE_PROFIT", size, 0.0, candles_held)

        return None

    def _make_trade(
        self,
        signal:      TradeSignal,
        df:          pd.DataFrame,
        entry_idx:   int,
        exit_idx:    int,
        exit_price:  float,
        exit_type:   str,
        size:        float,
        prior_tp_pnl: float,
        candles_held: int,
    ) -> BacktestTrade:
        entry = signal.entry_price
        if signal.direction == "LONG":
            pnl = (exit_price - entry) * size + prior_tp_pnl
        else:
            pnl = (entry - exit_price) * size + prior_tp_pnl

        # Deduct taker fees (0.1% entry + 0.1% exit)
        fee_entry = entry * size * TAKER_FEE_RATE
        fee_exit  = exit_price * size * TAKER_FEE_RATE
        pnl       = pnl - fee_entry - fee_exit

        pnl_pct = (pnl / (entry * (size + size if prior_tp_pnl else size))) * 100

        return BacktestTrade(
            symbol       = self.symbol,
            direction    = signal.direction,
            entry_time   = df.index[entry_idx],
            exit_time    = df.index[exit_idx],
            entry_price  = entry,
            exit_price   = round(exit_price, 2),
            size         = size,
            pnl          = round(pnl, 4),
            pnl_pct      = round(pnl_pct, 4),
            exit_type    = exit_type,
            probability  = signal.probability,
            candles_held = candles_held,
        )

    # ─────────────────────────────────────────────────────────────
    #  Report printer
    # ─────────────────────────────────────────────────────────────

    def print_report(self, r: BacktestResult) -> None:
        from rich.console import Console
        from rich.table   import Table
        from rich         import box

        c = Console()

        def sep():
            c.print("[bold blue]" + "═" * 65 + "[/]")

        sign     = "+" if r.total_pnl >= 0 else ""
        pnl_col  = "green" if r.total_pnl >= 0 else "red"

        sep()
        c.print(f"[bold cyan]  BACKTEST REPORT — {r.symbol}  {r.interval}[/]")
        sep()
        c.print(f"  [dim]Period   :[/]  {r.period_start.strftime('%Y-%m-%d %H:%M')}  →  {r.period_end.strftime('%Y-%m-%d %H:%M')}")
        c.print(f"  [dim]Candles  :[/]  {r.candles}  (warmup excluded)")
        c.print(f"  [dim]Threshold:[/]  LONG ≥ {self.threshold}%   SHORT ≤ {self.short_threshold}%")

        c.print(f"\n  [bold]── Performance ─────────────────────────────────────[/]")
        c.print(f"  Initial Capital  : [cyan]${r.initial_capital:>12,.2f}[/]")
        c.print(f"  Final Capital    : [cyan]${r.final_capital:>12,.2f}[/]")
        c.print(f"  Total P&L        : [{pnl_col}]{sign}${r.total_pnl:>10,.2f}  ({sign}{r.total_pnl_pct:.2f}%)[/]")
        c.print(f"  Max Drawdown     : [red]{r.max_drawdown_pct:.2f}%[/]")

        c.print(f"\n  [bold]── Trade Statistics ────────────────────────────────[/]")
        c.print(f"  Total Trades     : [bold]{r.total_trades}[/]")
        c.print(f"  Win Rate         : [yellow]{r.win_rate:.1f}%[/]  ({len(r.winning_trades)}W / {len(r.losing_trades)}L)")
        c.print(f"  Profit Factor    : [{'green' if r.profit_factor >= 1.5 else 'yellow'}]{r.profit_factor:.2f}[/]  [dim](≥1.5 is good)[/]")
        c.print(f"  Avg Win          : [green]+${r.avg_win:,.4f}[/]")
        c.print(f"  Avg Loss         : [red]${r.avg_loss:,.4f}[/]")
        c.print(f"  Best Trade       : [green]+${r.best_trade:,.4f}[/]")
        c.print(f"  Worst Trade      : [red]${r.worst_trade:,.4f}[/]")
        c.print(f"  Max Win Streak   : [green]{r.max_winning_streak}[/]")
        c.print(f"  Max Loss Streak  : [red]{r.max_losing_streak}[/]")
        c.print(f"  Avg Candles Held : {r.avg_candles_held:.1f}")

        # Exit type distribution
        from collections import Counter
        exits = Counter(t.exit_type for t in r.trades)
        c.print(f"\n  [bold]── Exit Distribution ───────────────────────────────[/]")
        for exit_type, count in exits.most_common():
            pct = count / r.total_trades * 100
            col = "green" if exit_type in ["TAKE_PROFIT_2", "BREAKEVEN", "TRAILING_STOP"] else "red"
            c.print(f"  [{col}]{exit_type:<20}[/]  {count:>3}  ({pct:.1f}%)")

        # Monthly returns
        c.print(f"\n  [bold]── Monthly Returns ─────────────────────────────────[/]")
        returns = r.monthly_returns
        if returns:
            for month, pct in sorted(returns.items()):
                m_col = "green" if pct >= 0 else "red"
                m_sign = "+" if pct >= 0 else ""
                c.print(f"  {month} : [{m_col}]{m_sign}{pct:.2f}%[/]")
        else:
            c.print("  [dim]Not enough data for monthly breakdown[/]")

        # Trade log table (last 20)
        c.print(f"\n  [bold]── Trade Log (last 20) ─────────────────────────────[/]")
        t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
        t.add_column("Entry Time",  width=18)
        t.add_column("Dir",         width=6)
        t.add_column("Entry",       justify="right", width=10)
        t.add_column("Exit",        justify="right", width=10)
        t.add_column("Type",        width=16)
        t.add_column("P&L",         justify="right", width=12)
        t.add_column("Prob",        justify="right", width=7)
        t.add_column("Candles",     justify="right", width=8)

        for trade in r.trades[-20:]:
            pnl_col2 = "green" if trade.pnl >= 0 else "red"
            sign2    = "+" if trade.pnl >= 0 else ""
            d_col    = "green" if trade.direction == "LONG" else "red"
            t.add_row(
                trade.entry_time.strftime("%m-%d %H:%M"),
                f"[{d_col}]{trade.direction}[/]",
                f"${trade.entry_price:>8,.2f}",
                f"${trade.exit_price:>8,.2f}",
                trade.exit_type.replace("_", " "),
                f"[{pnl_col2}]{sign2}${trade.pnl:>8,.4f}[/]",
                f"{trade.probability:.1f}%",
                str(trade.candles_held),
            )
        c.print(t)
        sep()
