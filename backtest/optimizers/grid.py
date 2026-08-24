# ─────────────────────────────────────────────────────────────────
#  backtest/optimizer.py
#
#  Grid search over threshold × interval × atr_sl_mult.
#
#  Strategy:
#    - Fetch OHLCV + compute indicators ONCE per (symbol, interval)
#    - Reuse preloaded DataFrames for every parameter combo
#    - Filter out combos with < MIN_TRADES (not statistically valid)
#    - Rank by Profit Factor (primary) → print top 15 + worst 5
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table   import Table
from rich         import box

from data.binance_client       import BinanceClient
from indicators                import add_all_indicators
from config.settings           import INITIAL_CAPITAL_USDT, MAX_RISK_PER_TRADE_PCT, MACRO_MAX_ADX, MACRO_SESSION_START, MACRO_SESSION_END

# ── Trading fees (Binance spot taker: 0.1% per side = 0.2% round-trip) ────────
TAKER_FEE_RATE = 0.001


# ─────────────────────────────────────────────────────────────────
#  Grid definition — edit these to expand/narrow the search
# ─────────────────────────────────────────────────────────────────

THRESHOLDS   = [65.0, 68.0, 70.0, 72.0, 75.0]      # LONG entry probability %
INTERVALS    = ["15m", "30m", "1h"]                # Candle timeframes
ATR_MULTS    = [0.75, 1.0, 1.25, 1.5]            # ATR × for stop-loss distance
MIN_TRADES   = 3                                   # Ignore combos with fewer trades (need statistical significance)
CANDLES      = 1000                                # Max candles per fetch (Binance limit)


# ─────────────────────────────────────────────────────────────────
#  Result record
# ─────────────────────────────────────────────────────────────────

@dataclass
class OptResult:
    symbol:      str
    interval:    str
    threshold:   float
    atr_mult:    float
    trades:      int
    win_rate:    float
    profit_factor: float
    total_pnl:   float
    total_pnl_pct: float
    max_dd:      float
    avg_candles: float

    @property
    def score(self) -> float:
        """
        Composite ranking score.

        Quality gates (hard fail):
          - Fewer than MIN_TRADES       → not statistically significant
          - Profit Factor < 1.2         → no edge (losing more than winning per dollar)
          - Win Rate < 48%              → strategy is a net loser

        Score = PF × WR  (pure quality, no trade-count bonus).
        The old log(trades) bonus was rewarding spammy low-threshold combos
        that had many mediocre trades rather than few high-quality ones.
        """
        if self.trades < MIN_TRADES:
            return -999.0
        if self.profit_factor < 1.2:
            return -999.0
        if self.win_rate < 48.0:
            return -999.0
        return self.profit_factor * (self.win_rate / 100.0)


# ─────────────────────────────────────────────────────────────────
#  Optimizer
# ─────────────────────────────────────────────────────────────────

class GridOptimizer:
    """
    Runs a grid search across threshold × interval × atr_sl_mult.

    Data is fetched once per (symbol, interval) and cached — so the
    Binance API is only hit len(INTERVALS) times, not 60 times.

    Usage:
        opt = GridOptimizer(symbol="BTCUSDT")
        results = opt.run()
        opt.print_report(results)
    """

    def __init__(
        self,
        symbol:     str   = "BTCUSDT",
        thresholds: list  = THRESHOLDS,
        intervals:  list  = INTERVALS,
        atr_mults:  list  = ATR_MULTS,
        candles:    int   = CANDLES,
        capital:    float = INITIAL_CAPITAL_USDT,
    ):
        self.symbol     = symbol
        self.thresholds = thresholds
        self.intervals  = intervals
        self.atr_mults  = atr_mults
        self.candles    = min(candles, 1000)
        self.capital    = capital
        self.client     = BinanceClient()
        self.console    = Console()

        # Cache: interval → preloaded+indicator DataFrame
        self._data_cache: dict[str, pd.DataFrame] = {}

    # ─────────────────────────────────────────────────────────────
    #  Data cache
    # ─────────────────────────────────────────────────────────────

    def _load_data(self, interval: str) -> pd.DataFrame:
        """Fetch + compute indicators for an interval, cached after first call. Pre-computes signals."""
        if interval not in self._data_cache:
            from data.cache import DataCache
            cache = DataCache()
            df = cache.load(self.symbol, interval)
            if df is None:
                logger.info(f"Cache miss. Fetching {self.candles} {interval} candles for {self.symbol}...")
                df = self.client.get_ohlcv(self.symbol, interval, self.candles)
                logger.info(f"Computing indicators for {interval}...")
                df = add_all_indicators(df)
                cache.save(df, self.symbol, interval)

            logger.info(f"Vectorizing probability scores for {len(df)} rows...")
            self._vectorize_scores(df)

            self._data_cache[interval] = df
            logger.success(f"Loaded {len(df)} rows for {interval} into optimizer memory")
        return self._data_cache[interval]

    def _vectorize_scores(self, df: pd.DataFrame) -> None:
        """Vectorized equivalent of SignalScorer to precompute probabilities instantly."""
        import numpy as np
        from config.profiles import get_profile
        
        prof = get_profile(self.symbol)
        price = df["close"].values
        atr   = np.where(np.isnan(df["atr"].values), price * 0.004, df["atr"].values)

        # Gate A
        sup_dist_abs = np.where(np.isnan(df["sr_support_dist_pct"].values), 9999.0, df["sr_support_dist_pct"].values * price / 100)
        res_dist_abs = np.where(np.isnan(df["sr_resist_dist_pct"].values), 9999.0, df["sr_resist_dist_pct"].values * price / 100)
        
        ga_long = np.where(sup_dist_abs <= atr * prof.sr_dist_near, 20.0, 
                  np.where(sup_dist_abs <= atr * prof.sr_dist_mid, 12.0, 
                  np.where(sup_dist_abs <= atr * prof.sr_dist_far, 5.0, 0.0)))
                  
        ga_short = np.where(res_dist_abs <= atr * prof.sr_dist_near, 20.0, 
                   np.where(res_dist_abs <= atr * prof.sr_dist_mid, 12.0, 
                   np.where(res_dist_abs <= atr * prof.sr_dist_far, 5.0, 0.0)))

        bb = df["bb_pct"].values
        bb_valid = ~np.isnan(bb)
        ga_long += np.where(bb_valid & (bb <= prof.bb_lower_extreme), 12.0, 
                   np.where(bb_valid & (bb <= prof.bb_lower_mid), 6.0, 0.0))
        ga_short += np.where(bb_valid & (bb >= prof.bb_upper_extreme), 12.0, 
                    np.where(bb_valid & (bb >= prof.bb_upper_mid), 6.0, 0.0))

        ps1 = df["pivot_s1"].values
        ps2 = df["pivot_s2"].values
        pr1 = df["pivot_r1"].values
        pr2 = df["pivot_r2"].values
        ga_long += np.where(~np.isnan(ps1) & (price <= ps1), 4.0, 0.0)
        ga_long += np.where(~np.isnan(ps2) & (price <= ps2), 4.0, 0.0)
        ga_short += np.where(~np.isnan(pr1) & (price >= pr1), 4.0, 0.0)
        ga_short += np.where(~np.isnan(pr2) & (price >= pr2), 4.0, 0.0)

        # Gate B
        gb_long, gb_short = np.zeros(len(df)), np.zeros(len(df))
        rsi = df["rsi"].values
        rsi_v = ~np.isnan(rsi)
        gb_long += np.where(rsi_v & (rsi <= prof.rsi_oversold_extreme), 15.0,
                   np.where(rsi_v & (rsi <= prof.rsi_oversold_mid), 8.0, 0.0))
        gb_short += np.where(rsi_v & (rsi >= prof.rsi_overbought_extreme), 15.0,
                    np.where(rsi_v & (rsi >= prof.rsi_overbought_mid), 8.0, 0.0))

        st_k = df["stoch_rsi_k"].values
        st_d = df["stoch_rsi_d"].values
        st_v = ~(np.isnan(st_k) | np.isnan(st_d))
        gb_long += np.where(st_v & (st_k <= prof.stoch_oversold_extreme) & (st_k > st_d), 10.0,
                   np.where(st_v & (st_k <= prof.stoch_oversold_mid), 5.0, 0.0))
        gb_short += np.where(st_v & (st_k >= prof.stoch_overbought_extreme) & (st_k < st_d), 10.0,
                    np.where(st_v & (st_k >= prof.stoch_overbought_mid), 5.0, 0.0))

        macd = df["macd_hist"].values
        mac_v = ~np.isnan(macd)
        gb_long += np.where(mac_v & (macd > 0), 10.0,
                   np.where(mac_v & (macd > -atr * prof.macd_near_zero), 5.0, 0.0))
        gb_short += np.where(mac_v & (macd < 0), 10.0,
                    np.where(mac_v & (macd < atr * prof.macd_near_zero), 5.0, 0.0))

        # Gate C
        gc_long, gc_short = np.zeros(len(df)), np.zeros(len(df))
        sma = df["sma_200"].values
        sma_v = ~np.isnan(sma)
        gc_long += np.where(sma_v & (price > sma), 10.0, 0.0)
        gc_short += np.where(sma_v & (price < sma), 10.0, 0.0)

        vwap = df["vwap"].values
        vwap_v = ~np.isnan(vwap)
        gc_long += np.where(vwap_v & (price > vwap), 7.0, 0.0)
        gc_short += np.where(vwap_v & (price < vwap), 7.0, 0.0)

        vol = df["volume_ratio"].values
        vol_v = ~np.isnan(vol)
        gc_long += np.where(vol_v & (vol >= prof.vol_ratio_surge), 8.0,
                   np.where(vol_v & (vol >= prof.vol_ratio_high), 4.0, 0.0))
        gc_short += np.where(vol_v & (vol >= prof.vol_ratio_surge), 8.0,
                    np.where(vol_v & (vol >= prof.vol_ratio_high), 4.0, 0.0))

        tot_long = ga_long + gb_long + gc_long
        tot_short = ga_short + gb_short + gc_short

        # ── ADX Regime Filter ──────────────────────────────────────
        adx = df["adx"].values

        regime_valid = ~np.isnan(adx)
        safe_regime = regime_valid & (adx <= MACRO_MAX_ADX)

        # Kill signals outside safe regime
        tot_long = np.where(safe_regime, tot_long, 0.0)
        tot_short = np.where(safe_regime, tot_short, 0.0)

        # ── Frozen Rule 2 & 3: Time and Day Blocks ───────────────────────────
        # Discovered on BTC: Only trade during Asian open / late US (16:00 - 23:59 UTC)
        # and NEVER on Tuesday.
        hour_of_day = df.index.hour
        day_of_week = df.index.weekday  # 0=Mon, 1=Tue, ..., 6=Sun

        # We want to ALLOW hours MACRO_SESSION_START to MACRO_SESSION_END-1
        valid_hour = (hour_of_day >= MACRO_SESSION_START) & (hour_of_day < MACRO_SESSION_END)
        valid_day  = (day_of_week != 0) & (day_of_week != 1)  # Mon+Tue excluded — PF<0.80
        
        # If not valid, zero out the signal
        invalid_time = ~(valid_hour & valid_day)
        
        tot_long  = np.where(invalid_time, 0.0, tot_long)
        tot_short = np.where(invalid_time, 0.0, tot_short)
        # ─────────────────────────────────────────────────────────────────────

        is_long = tot_long >= tot_short
        
        prob = np.where(is_long, tot_long, 100.0 - tot_short)
        
        df["_fast_prob"] = prob
        df["_fast_is_long"] = is_long

    # ─────────────────────────────────────────────────────────────
    #  Single combo runner
    # ─────────────────────────────────────────────────────────────

    def _run_combo(
        self,
        df:         pd.DataFrame,
        interval:   str,
        threshold:  float,
        atr_mult:   float,
    ) -> OptResult:
        """Run one parameter combination — All-In All-Out matching BacktestEngine."""
        import numpy as np
        WARMUP = 200

        probs    = df["_fast_prob"].values
        is_longs = df["_fast_is_long"].values
        closes   = df["close"].values
        highs    = df["high"].values
        lows     = df["low"].values

        raw_atrs = df["atr"].values
        atrs     = np.where(np.isnan(raw_atrs), closes * 0.004, raw_atrs)

        cash        = self.capital
        trades_pnl  = []
        i           = WARMUP
        n_rows      = len(df)

        while i < n_rows - 1:
            prob = probs[i]

            if prob >= threshold and is_longs[i]:
                entry   = closes[i]
                atr     = atrs[i]
                sl_dist = atr * atr_mult

                if sl_dist == 0:
                    i += 1
                    continue

                new_sl = entry - sl_dist
                new_tp = entry + sl_dist * 1.5

                size = (cash * (MAX_RISK_PER_TRADE_PCT / 100)) / sl_dist
                trade_pnl = None
                exit_idx  = i

                for j in range(i + 1, n_rows):
                    cl = lows[j]
                    ch = highs[j]

                    if cl <= new_sl:
                        trade_pnl = (new_sl - entry) * size
                        trade_pnl -= (entry + new_sl) * size * TAKER_FEE_RATE
                        exit_idx = j
                        break

                    if ch >= new_tp:
                        trade_pnl = (new_tp - entry) * size
                        trade_pnl -= (entry + new_tp) * size * TAKER_FEE_RATE
                        exit_idx = j
                        break

                if trade_pnl is not None:
                    cash += trade_pnl
                    trades_pnl.append(trade_pnl)
                    i = exit_idx + 1
                    continue

            i += 1

        # Compute metrics
        trades_arr = np.array(trades_pnl)
        wins   = trades_arr[trades_arr > 0]
        losses = trades_arr[trades_arr <= 0]
        gross_profit = wins.sum() if len(wins) > 0 else 0.0
        gross_loss   = abs(losses.sum()) if len(losses) > 0 else 0.0

        pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        wr = (len(wins) / len(trades_arr) * 100) if len(trades_arr) > 0 else 0.0

        # Max drawdown
        cap, peak, max_dd = self.capital, self.capital, 0.0
        for pnl in trades_pnl:
            cap  += pnl
            if cap > peak: peak = cap
            dd = (peak - cap) / peak * 100
            if dd > max_dd: max_dd = dd

        return OptResult(
            symbol        = self.symbol,
            interval      = interval,
            threshold     = threshold,
            atr_mult      = atr_mult,
            trades        = len(trades_pnl),
            win_rate      = round(wr, 1),
            profit_factor = pf,
            total_pnl     = round(cash - self.capital, 2),
            total_pnl_pct = round((cash - self.capital) / self.capital * 100, 2),
            max_dd        = round(max_dd, 2),
            avg_candles   = 0.0,
        )

    # ─────────────────────────────────────────────────────────────
    #  Main runner
    # ─────────────────────────────────────────────────────────────

    def run(self, external_df: pd.DataFrame = None, external_interval: str = None) -> list[OptResult]:
        if external_df is not None and external_interval is not None:
            intervals_to_run = [external_interval]
        else:
            intervals_to_run = self.intervals

        combos = list(itertools.product(intervals_to_run, self.thresholds, self.atr_mults))
        total  = len(combos)

        # Only print the big header if we are not suppressing output (which we might want for WFO, but we'll leave it for now)
        self.console.print(
            f"\n[bold cyan]🔍 Grid Search  ─  {self.symbol}[/]\n"
            f"[dim]  Thresholds : {self.thresholds}\n"
            f"  Intervals  : {intervals_to_run}\n"
            f"  ATR mults  : {self.atr_mults}\n"
            f"  Total combos: {total}  (data fetched once per interval)[/]\n"
        )

        results: list[OptResult] = []

        for idx, (interval, threshold, atr_mult) in enumerate(combos, 1):
            self.console.print(
                f"[dim]  [{idx:>3}/{total}]  {interval}  threshold={threshold:.0f}%  "
                f"atr={atr_mult:.2f}×[/]",
                end="\r",
            )
            
            if external_df is not None and interval == external_interval:
                df = external_df
                if "_fast_prob" not in df.columns:
                    self._vectorize_scores(df)
            else:
                df = self._load_data(interval)
                
            res = self._run_combo(df, interval, threshold, atr_mult)
            results.append(res)

        self.console.print()   # newline after the \r progress line
        return results

    # ─────────────────────────────────────────────────────────────
    #  Report
    # ─────────────────────────────────────────────────────────────

    def print_report(self, results: list[OptResult]) -> None:
        c = self.console

        # Filter: minimum trade count
        valid   = [r for r in results if r.trades >= MIN_TRADES]
        invalid = [r for r in results if r.trades <  MIN_TRADES]

        # Sort by composite score
        valid.sort(key=lambda r: r.score, reverse=True)

        def sep(title=""):
            line = "═" * 70
            c.print(f"\n[bold blue]{line}[/]")
            if title:
                c.print(f"[bold cyan]  {title}[/]")

        sep(f"GRID SEARCH RESULTS — {self.symbol}")
        c.print(
            f"  [dim]Total combos: {len(results)}   "
            f"Valid (≥{MIN_TRADES} trades): {len(valid)}   "
            f"Filtered out: {len(invalid)}[/]"
        )

        def make_table(title: str, rows: list[OptResult], highlight_top: bool = False) -> Table:
            t = Table(
                title=title,
                box=box.SIMPLE_HEAD,
                show_header=True,
                header_style="bold cyan",
                title_style="bold yellow",
            )
            t.add_column("Rank",      width=5,  justify="right")
            t.add_column("Interval",  width=9)
            t.add_column("Thresh",    width=8,  justify="right")
            t.add_column("ATR Mult",  width=9,  justify="right")
            t.add_column("Trades",    width=7,  justify="right")
            t.add_column("WR %",      width=7,  justify="right")
            t.add_column("PF",        width=7,  justify="right")
            t.add_column("P&L %",     width=9,  justify="right")
            t.add_column("Max DD %",  width=9,  justify="right")

            for rank, r in enumerate(rows, 1):
                pf_str    = f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "∞"
                pf_col    = "green" if r.profit_factor >= 1.5 else ("yellow" if r.profit_factor >= 1.0 else "red")
                pnl_col   = "green" if r.total_pnl_pct >= 0 else "red"
                sign      = "+" if r.total_pnl_pct >= 0 else ""

                r_str  = str(rank)
                iv_str = str(r.interval)
                th_str = f"{r.threshold:.0f}%"
                am_str = f"{r.atr_mult:.2f}×"
                tr_str = str(r.trades)
                wr_str = f"{r.win_rate:.1f}%"
                pf_fmt = f"[{pf_col}]{pf_str}[/]"
                pl_fmt = f"[{pnl_col}]{sign}{r.total_pnl_pct:.2f}%[/]"
                dd_fmt = f"[red]{r.max_dd:.2f}%[/]"

                if highlight_top and rank == 1:
                    r_str  = f"[bold]{r_str}[/]"
                    iv_str = f"[bold]{iv_str}[/]"
                    th_str = f"[bold]{th_str}[/]"
                    am_str = f"[bold]{am_str}[/]"
                    tr_str = f"[bold]{tr_str}[/]"
                    wr_str = f"[bold]{wr_str}[/]"
                    pf_fmt = f"[bold]{pf_fmt}[/]"
                    pl_fmt = f"[bold]{pl_fmt}[/]"
                    dd_fmt = f"[bold]{dd_fmt}[/]"

                t.add_row(r_str, iv_str, th_str, am_str, tr_str, wr_str, pf_fmt, pl_fmt, dd_fmt)
            return t

        # Top 15
        if valid:
            c.print()
            c.print(make_table("🏆  TOP 15 PARAMETER COMBOS  (ranked by PF × WR × log(trades))", valid[:15], highlight_top=True))

            # Best combo summary
            best = valid[0]
            sep("🥇 BEST COMBO")
            c.print(f"  Interval  : [bold cyan]{best.interval}[/]")
            c.print(f"  Threshold : [bold cyan]{best.threshold:.0f}%[/]")
            c.print(f"  ATR Mult  : [bold cyan]{best.atr_mult:.2f}×[/]")
            c.print(f"  Trades    : [bold]{best.trades}[/]")
            c.print(f"  Win Rate  : [yellow]{best.win_rate:.1f}%[/]")
            pf_str = f"{best.profit_factor:.2f}" if best.profit_factor != float("inf") else "∞"
            c.print(f"  Prof Factor: [green]{pf_str}[/]")
            sign = "+" if best.total_pnl_pct >= 0 else ""
            c.print(f"  Total P&L  : [{'green' if best.total_pnl >= 0 else 'red'}]{sign}${best.total_pnl:,.2f}  ({sign}{best.total_pnl_pct:.2f}%)[/]")
            c.print(f"  Max DD     : [red]{best.max_dd:.2f}%[/]")
        else:
            c.print("\n[red]  ⚠️  No combos had ≥ {MIN_TRADES} trades.[/]")

        # Worst 5
        if len(valid) > 5:
            worst = valid[-5:][::-1]
            c.print()
            c.print(make_table("⚠️  WORST 5 (avoid these)", worst))

        # Filtered out summary
        if invalid:
            sep(f"FILTERED OUT — {len(invalid)} combos had < {MIN_TRADES} trades")
            for r in invalid[:10]:
                c.print(f"  [dim]{r.interval}  thresh={r.threshold:.0f}%  atr={r.atr_mult:.2f}×  →  {r.trades} trade(s)[/]")
            if len(invalid) > 10:
                c.print(f"  [dim]... and {len(invalid) - 10} more[/]")

        sep()
