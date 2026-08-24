# ─────────────────────────────────────────────────────────────────
#  backtest/engines/walk_forward.py
#
#  Walk-Forward Optimization Engine.
#
#  Methodology:
#    - Slices historical data into rolling windows.
#    - FOLD: Train on `train_days`, Test on `test_days`.
#    - Finds the best parameters in the Train window using GridOptimizer.
#    - Trades those parameters in the Test window using BacktestEngine.
#    - Rolls the window forward by `test_days`.
#    - Stitches all Test window trades into a single BacktestResult.
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations
from datetime import timedelta

import pandas as pd
from loguru import logger
from rich.console import Console

from data.binance_client       import BinanceClient
from data.cache                import DataCache
from indicators                import add_all_indicators
from backtest.optimizers.grid  import GridOptimizer
from backtest.engines.standard import BacktestEngine, BacktestResult, BacktestTrade
from config.settings           import INITIAL_CAPITAL_USDT


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization Engine.
    """

    def __init__(
        self,
        symbol:     str = "BTCUSDT",
        interval:   str = "1h",
        candles:    int = 87000,
        train_days: int = 180,
        test_days:  int = 60,
        capital:    float = INITIAL_CAPITAL_USDT,
    ):
        self.symbol     = symbol
        self.interval   = interval
        self.candles    = candles
        self.train_days = train_days
        self.test_days  = test_days
        self.capital    = capital
        self.client     = BinanceClient()
        self.cache      = DataCache()
        self.console    = Console()

    def _load_raw_data(self) -> pd.DataFrame:
        """Load raw OHLCV without indicators. Indicators computed per-fold to prevent data leakage."""
        df = self.cache.load(self.symbol, self.interval)
        if df is not None:
            indicator_cols = [c for c in df.columns if c not in ("open", "high", "low", "close", "volume")]
            if indicator_cols:
                df = df.drop(columns=indicator_cols)
            return df
        logger.info(f"Cache miss. Fetching {self.candles} {self.interval} candles for {self.symbol}...")
        df = self.client.get_ohlcv(self.symbol, self.interval, self.candles)
        self.cache.save(df, self.symbol, self.interval)
        return df

    def run(self) -> BacktestResult:
        df_raw = self._load_raw_data()

        if not isinstance(df_raw.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        start_time = df_raw.index[0]
        end_time   = df_raw.index[-1]

        current_train_start = start_time

        all_oos_trades: list[BacktestTrade] = []
        current_capital = self.capital

        fold = 1

        WARMUP_ROWS = 210

        self.console.print(f"\n[bold magenta]🚀 Walk-Forward Optimization — {self.symbol}[/]")
        self.console.print(f"[dim]Train: {self.train_days} days | Test: {self.test_days} days[/]\n")

        while True:
            train_end = current_train_start + timedelta(days=self.train_days)
            test_end  = train_end + timedelta(days=self.test_days)

            if train_end > end_time:
                break

            # ── Slice raw OHLCV, then compute indicators per-fold ──
            df_train_raw = df_raw[(df_raw.index >= current_train_start) & (df_raw.index < train_end)]
            df_train = add_all_indicators(df_train_raw)

            # Prepend WARMUP_ROWS from training end for test slice indicator warmup
            df_test_warmup_raw = df_raw[df_raw.index < train_end].iloc[-WARMUP_ROWS:]
            df_test_body_raw   = df_raw[(df_raw.index >= train_end) & (df_raw.index < test_end)]
            df_test_combined   = pd.concat([df_test_warmup_raw, df_test_body_raw])
            df_test            = add_all_indicators(df_test_combined)

            if len(df_train) < 500 or len(df_test_body_raw) < 10:
                logger.warning(f"Fold {fold}: Not enough data. Skipping.")
                break

            self.console.print(
                f"[bold cyan]Fold {fold}[/] | "
                f"Train: {current_train_start.strftime('%Y-%m-%d')} → {train_end.strftime('%Y-%m-%d')} "
                f"({len(df_train)} candles) | "
                f"Test: → {test_end.strftime('%Y-%m-%d')} ({len(df_test_body_raw)} candles)"
            )

            # ── 1. OPTIMIZE (In-Sample) ───────────────────────────
            # Fix #1 — use thresholds in the 60-75% range where the
            # signal scorer actually has selectivity (not every candle qualifies).
            opt = GridOptimizer(
                symbol=self.symbol,
                capital=current_capital,
                thresholds=[46.0, 48.0, 50.0, 52.0, 54.0],
                atr_mults=[0.75, 1.0, 1.25, 1.5],
            )

            with self.console.capture():
                results = opt.run(external_df=df_train, external_interval=self.interval)

            # Quality gate: require real edge in training, not just frequency luck
            valid_results = [r for r in results if r.score > -999.0]
            valid_results.sort(key=lambda r: r.score, reverse=True)

            if not valid_results:
                self.console.print(
                    "  [yellow]⚠ No combo passed quality gates (PF≥1.2, WR≥48%, trades≥5) "
                    "in training window. Skipping test fold.[/]"
                )
                current_train_start += timedelta(days=self.test_days)
                fold += 1
                continue

            best = valid_results[0]
            self.console.print(
                f"  [dim]Best Params:[/] Threshold={best.threshold:.0f}%  "
                f"ATR={best.atr_mult:.2f}×  "
                f"(Train PF={best.profit_factor:.2f}  WR={best.win_rate:.1f}%  "
                f"Trades={best.trades})"
            )

            # ── 2. TEST (Out-Of-Sample) ───────────────────────────
            # Fix #2 — disable SHORTs in the test engine so it runs
            # LONG-only, identical to what the GridOptimizer trained on.
            engine = BacktestEngine(
                symbol=self.symbol,
                interval=self.interval,
                threshold=best.threshold,
                short_threshold=0.0,          # LONG-only — matches GridOptimizer training
                atr_sl_mult=best.atr_mult,
                capital=current_capital,
            )

            test_result = engine.run(external_df=df_test)

            # Filter to trades that opened inside the true test window
            # (engine.run WARMUP skips the pre-pended training rows so this
            # filter should match perfectly, but we keep it as a safety net)
            test_trades = [t for t in test_result.trades if t.entry_time >= train_end]

            pnl = sum(t.pnl for t in test_trades)
            col = "green" if pnl >= 0 else "red"
            sign = "+" if pnl >= 0 else ""
            self.console.print(
                f"  [dim]Test Result:[/] [{col}]{len(test_trades)} trades  "
                f"P&L: {sign}${pnl:.2f}[/]\n"
            )

            all_oos_trades.extend(test_trades)
            current_capital += pnl

            current_train_start += timedelta(days=self.test_days)
            fold += 1

        # ── Compile final stitched result ─────────────────────────
        final_result = BacktestResult(
            symbol=self.symbol,
            interval=self.interval,
            period_start=df_raw.index[0],
            period_end=df_raw.index[-1],
            candles=len(df_raw),
            initial_capital=self.capital,
            final_capital=current_capital,
        )
        final_result.trades = all_oos_trades

        total_folds = fold - 1
        self.console.print(
            f"\n[bold magenta]✅ WFO Complete — {total_folds} folds  |  "
            f"{len(all_oos_trades)} stitched OOS trades[/]"
        )

        engine = BacktestEngine(symbol=self.symbol)
        engine.print_report(final_result)

        return final_result

