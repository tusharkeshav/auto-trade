# ─────────────────────────────────────────────────────────────────
#  backtest/engines/master_portfolio.py
#  Institutional Shared-Book Portfolio Backtest Engine.
#
#  Mathematical & Architectural Basis:
#    - Simulates true institutional portfolio dynamics: shared capital
#      pool, concentration caps (MAX_POSITIONS_PER_MASTER), and signal
#      conviction prioritization.
#    - Mirrors production daemon (india_paper_trade.py) 2-phase daily logic:
#        Phase 1: Position management & exits first (freeing capital/slots)
#        Phase 2: Probability-sorted entries second (highest score wins slot)
#    - Zero leverage, pure CNC Delivery cash sizing.
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from backtest.engines.master_to_stock import MasterStockTrade, MasterStockResult
from config.india_settings import INITIAL_CAPITAL_INR, INDIA_MAX_RISK_PER_TRADE_PCT, INDIA_SIGNAL_THRESHOLD
from data.india.nse_client import NSEClient
from indicators import add_all_indicators
from probability.unified_cross_scorer import UnifiedCrossScorer
from probability.hmm_multiplexer      import HMMStrategyMultiplexer
from engine.india_costs import calculate_round_trip_cost
from ml.meta_labeler import XGBoostMetaLabeler, add_hmm_regime_features


@dataclass
class PortfolioPosition:
    symbol:        str
    master_index:  str
    entry_idx:     int
    entry_time:    datetime
    entry_price:   float
    qty:           float
    sl:            float
    tp:            float
    sl_dist:       float
    direction:     str
    setup_label:   str
    index_prob:    float
    features:      dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioResult:
    interval:          str
    period_start:      datetime
    period_end:        datetime
    candles:           int
    initial_capital:   float
    trades:            list[MasterStockTrade] = field(default_factory=list)
    final_capital:     float = 0.0
    stock_results:     dict[str, MasterStockResult] = field(default_factory=dict)

    @property
    def total_trades(self) -> int: return len(self.trades)

    @property
    def winning_trades(self) -> list[MasterStockTrade]: return [t for t in self.trades if t.net_pnl > 0]

    @property
    def losing_trades(self) -> list[MasterStockTrade]: return [t for t in self.trades if t.net_pnl <= 0]

    @property
    def win_rate(self) -> float:
        return len(self.winning_trades) / self.total_trades * 100 if self.trades else 0.0

    @property
    def total_pnl(self) -> float: return self.final_capital - self.initial_capital

    @property
    def total_pnl_pct(self) -> float: return self.total_pnl / self.initial_capital * 100

    @property
    def profit_factor(self) -> float:
        gp = sum(t.net_pnl for t in self.winning_trades)
        gl = abs(sum(t.net_pnl for t in self.losing_trades))
        return round(gp / gl, 2) if gl else float("inf")

    @property
    def total_costs_inr(self) -> float: return sum(t.cost_inr for t in self.trades)


class MasterPortfolioEngine:
    """
    Simulates multi-stock shared capital pool execution with signal priority and concentration guard.
    """

    def __init__(
        self,
        routing_map:  Dict[str, str],
        interval:     str = "1d",
        bars:         int = 500,
        threshold:    float = INDIA_SIGNAL_THRESHOLD,
        atr_sl_mult:  float = 1.0,
        atr_tp_mult:  float = 3.0,
        capital:      float = INITIAL_CAPITAL_INR,
        vix:          float = 16.0,
        max_positions_per_master: int = 1,
        max_open_trades: int = 8,
    ):
        self.routing_map              = routing_map
        self.interval                 = interval
        self.bars                     = bars
        self.threshold                = threshold
        self.atr_sl_mult              = atr_sl_mult
        self.atr_tp_mult              = atr_tp_mult
        self.capital                  = capital
        self.vix                      = vix
        self.max_positions_per_master = max_positions_per_master
        self.max_open_trades          = max_open_trades
        self.quarantine_bars: Dict[str, int] = {stk: 0 for stk in routing_map.keys()}
        self.ev_window: int = 10         # Look at last 10 trades
        self.ev_min_trades: int = 999    # Disabled by default in favor of 60-Day RS Gate
        self.ev_min_wr: float = 40.0     # Minimum 40% Win Rate required
        self.quarantine_cooldown: int = 20 # Quarantine for 20 daily bars (~1 month)
        self.client                   = NSEClient()
        self.meta_labeler             = XGBoostMetaLabeler(min_train_trades=15)
        self.scorers: Dict[str, HMMStrategyMultiplexer] = {}

        for master in set(routing_map.values()):
            self.scorers[master] = HMMStrategyMultiplexer(
                symbol      = master,
                threshold   = threshold,
                atr_sl_mult = atr_sl_mult,
                atr_tp_mult = atr_tp_mult,
                interval    = interval,
                current_vix = vix,
            )

    def run(self) -> PortfolioResult:
        logger.info(f"Fetching {self.bars} × {self.interval} candles for Portfolio Universe ({len(self.routing_map)} stocks)...")

        # 1. Fetch all data and apply indicators
        master_dfs: Dict[str, pd.DataFrame] = {}
        stock_dfs: Dict[str, pd.DataFrame] = {}

        for master in set(self.routing_map.values()):
            master_dfs[master] = add_all_indicators(self.client.get_ohlcv(master, self.interval, self.bars))

        if "^NSEI" not in master_dfs:
            master_dfs["^NSEI"] = add_all_indicators(self.client.get_ohlcv("^NSEI", self.interval, self.bars))

        for stk in self.routing_map.keys():
            stock_dfs[stk] = add_all_indicators(self.client.get_ohlcv(stk, self.interval, self.bars))

        # 2. Align timestamps by intersection across all dataframes
        common_idx = None
        for df in list(master_dfs.values()) + list(stock_dfs.values()):
            if common_idx is None:
                common_idx = df.index
            else:
                common_idx = common_idx.intersection(df.index)

        common_idx = common_idx.sort_values()
        if len(common_idx) < 50:
            raise ValueError(f"Insufficient common aligned candles across universe: {len(common_idx)}")

        for m in master_dfs: master_dfs[m] = add_hmm_regime_features(master_dfs[m].loc[common_idx])
        for s in stock_dfs:  stock_dfs[s]  = stock_dfs[s].loc[common_idx]

        WARMUP = min(200, len(common_idx) // 4)

        result = PortfolioResult(
            interval        = self.interval,
            period_start    = common_idx[WARMUP],
            period_end      = common_idx[-1],
            candles         = len(common_idx) - WARMUP,
            initial_capital = self.capital,
            final_capital   = self.capital,
        )

        # Initialize per-stock results for detailed reporting
        for stk, master in self.routing_map.items():
            result.stock_results[stk] = MasterStockResult(
                symbol          = stk,
                master_index    = master,
                interval        = self.interval,
                period_start    = common_idx[WARMUP],
                period_end      = common_idx[-1],
                candles         = len(common_idx) - WARMUP,
                initial_capital = self.capital / len(self.routing_map),
                final_capital   = self.capital / len(self.routing_map),
            )

        cash = self.capital
        positions: Dict[str, PortfolioPosition] = {}
        all_trades: List[MasterStockTrade] = []

        # 3. Step chronologically through candles
        for i in range(WARMUP, len(common_idx)):
            current_time = common_idx[i].to_pydatetime()

            # ── Phase 1: Manage Existing Positions (Exits First) ──
            closed_symbols = []
            for stk, p in positions.items():
                row_stk = stock_dfs[stk].iloc[i]
                high, low = float(row_stk["high"]), float(row_stk["low"])
                exit_type = None
                exit_price = 0.0

                if low <= p.sl:
                    exit_type = "TRAIL_STOP" if p.sl >= p.entry_price else "STOP_LOSS"
                    exit_price = p.sl
                elif high >= p.tp:
                    exit_type = "TAKE_PROFIT"
                    exit_price = p.tp
                else:
                    # Trailing stop matching +1.5R lock-in / +1.0R break-even
                    if high >= p.entry_price + p.sl_dist * 1.5 and p.sl < p.entry_price + p.sl_dist * 0.5:
                        p.sl = round(p.entry_price + p.sl_dist * 0.5, 2)
                    elif high >= p.entry_price + p.sl_dist * 1.0 and p.sl < p.entry_price:
                        p.sl = round(p.entry_price, 2)

                # Timeout check on last candle
                if not exit_type and i == len(common_idx) - 1:
                    exit_type = "TIMEOUT"
                    exit_price = float(row_stk["close"])

                if exit_type:
                    candles_held = i - p.entry_idx
                    gross_pnl = (exit_price - p.entry_price) * p.qty
                    buy_cost, sell_cost = calculate_round_trip_cost(p.entry_price, exit_price, p.qty, "CNC")
                    cost_inr = buy_cost.total + sell_cost.total
                    net_pnl = gross_pnl - cost_inr
                    net_pct = net_pnl / (p.entry_price * p.qty) * 100 if (p.entry_price * p.qty) > 0 else 0.0

                    trade = MasterStockTrade(
                        symbol       = stk,
                        master_index = p.master_index,
                        entry_time   = p.entry_time,
                        exit_time    = current_time,
                        direction    = p.direction,
                        entry_price  = round(p.entry_price, 2),
                        exit_price   = round(exit_price, 2),
                        qty          = p.qty,
                        gross_pnl    = round(gross_pnl, 2),
                        cost_inr     = round(cost_inr, 2),
                        net_pnl      = round(net_pnl, 2),
                        net_pnl_pct  = round(net_pct, 4),
                        exit_type    = exit_type,
                        setup_label  = p.setup_label,
                        index_prob   = p.index_prob,
                        candles_held = candles_held,
                        features     = getattr(p, "features", {}),
                    )

                    cash += (p.entry_price * p.qty) + net_pnl
                    all_trades.append(trade)
                    result.stock_results[stk].trades.append(trade)
                    closed_symbols.append(stk)

            for sym in closed_symbols:
                del positions[sym]

            # ── Phase 2: Prioritized New Entries ──
            for stk in self.quarantine_bars:
                if self.quarantine_bars[stk] > 0:
                    self.quarantine_bars[stk] -= 1

            candidate_entries = []
            for stk, master in self.routing_map.items():
                if stk in positions:
                    continue

                if self.quarantine_bars[stk] > 0:
                    continue

                if i >= 60 and "^NSEI" in master_dfs:
                    stk_close = float(stock_dfs[stk].iloc[i]["close"])
                    stk_close_60 = float(stock_dfs[stk].iloc[i - 60]["close"])

                    nifty_close = float(master_dfs["^NSEI"].iloc[i]["close"])
                    nifty_close_60 = float(master_dfs["^NSEI"].iloc[i - 60]["close"])

                    if nifty_close > 0 and nifty_close_60 > 0 and stk_close_60 > 0:
                        rs_today = stk_close / nifty_close
                        rs_60 = stk_close_60 / nifty_close_60
                        rs_slope = ((rs_today - rs_60) / rs_60) * 100.0

                        # GATE: If stock is underperforming NIFTY50 over 60 bars (~3 months), block entry!
                        if rs_slope <= 0.0:
                            continue

                stk_trades = [t for t in all_trades if t.symbol == stk][-self.ev_window:]
                if len(stk_trades) >= self.ev_min_trades:
                    wins = sum(1 for t in stk_trades if t.net_pnl > 0)
                    wr = (wins / len(stk_trades)) * 100.0
                    if wr < self.ev_min_wr:
                        self.quarantine_bars[stk] = self.quarantine_cooldown
                        continue

                row_idx   = master_dfs[master].iloc[i]
                slice_idx = master_dfs[master].iloc[: i + 1]
                scorer    = self.scorers[master]

                signal = scorer.score(row_idx, slice_idx)
                if signal.is_tradeable() and signal.direction == "LONG":
                    row_stk = stock_dfs[stk].iloc[i]
                    ltp = float(row_stk["close"])
                    candidate_entries.append((signal.probability, stk, master, ltp, row_stk, signal, i))

            # Sort descending by signal probability
            candidate_entries.sort(key=lambda x: x[0], reverse=True)

            for prob, stk, master, ltp, row_stk, signal, idx in candidate_entries:
                # Check concentration limit across shared book
                if len(positions) >= self.max_open_trades:
                    break

                master_counts = sum(1 for p in positions.values() if p.master_index == master)
                if master_counts >= self.max_positions_per_master:
                    continue

                row_idx = master_dfs[master].iloc[idx]
                rs_today = ltp / float(row_idx["close"]) if float(row_idx["close"]) > 0 else 1.0
                idx_60 = max(0, idx - 60)
                rs_60 = float(stock_dfs[stk].iloc[idx_60]["close"]) / float(master_dfs[master].iloc[idx_60]["close"]) if float(master_dfs[master].iloc[idx_60]["close"]) > 0 else 1.0
                rs_slope = ((rs_today - rs_60) / rs_60) * 100.0 if idx >= 60 else 0.0
                atr_stk = row_stk.get("atr") if hasattr(row_stk, "get") else row_stk["atr"]
                if math.isnan(atr_stk) or atr_stk <= 0: atr_stk = ltp * 0.015

                hmm_state = int(row_idx.get("hmm_state", 0))
                p2 = float(row_idx.get("hmm_prob_2", 0.0))
                scalar = round(max(0.40, 1.0 - p2), 2)  # Sizing smoothly dampened by HMM crash prob (max shield 0.8% risk)

                feats = {
                    "hmm_state": float(hmm_state),
                    "hmm_crash_score": float(row_idx.get("hmm_crash_score", 0.0)),
                    "rs_slope_60d": float(rs_slope),
                    "adx_14": float(row_idx.get("adx", 20.0)),
                    "vix": float(row_idx.get("vix", 16.0)),
                    "rsi_14": float(row_stk.get("rsi", 50.0)),
                    "atr_pct": float(atr_stk / ltp * 100.0) if ltp > 0 else 1.5,
                }

                sl_dist = atr_stk * self.atr_sl_mult
                sl = round(ltp - sl_dist, 2)
                tp = round(ltp + sl_dist * self.atr_tp_mult, 2)

                # Risk sizing based on shared portfolio capital * HMM adaptive scalar
                risk_inr = (self.capital * INDIA_MAX_RISK_PER_TRADE_PCT / 100) * scalar
                qty = risk_inr / sl_dist if sl_dist > 0 else 0
                max_affordable = cash / ltp if cash > 0 and ltp > 0 else 0
                qty = min(qty, max_affordable)

                if qty >= 1:
                    cost = ltp * qty
                    cash -= cost

                    reason_str = signal.reason or ""
                    if "Connors RSI-2" in reason_str:
                        setup_lbl = "CONNORS_MR"
                    elif "DUAL CONFIRM" in reason_str:
                        setup_lbl = "DUAL_CONFIRM"
                    elif "BAND BOUNCE" in reason_str:
                        setup_lbl = "BAND_BOUNCE"
                    else:
                        setup_lbl = "PULLBACK"

                    pos = PortfolioPosition(
                        symbol       = stk,
                        master_index = master,
                        entry_idx    = idx,
                        entry_time   = current_time,
                        entry_price  = ltp,
                        qty          = qty,
                        sl           = sl,
                        tp           = tp,
                        sl_dist      = sl_dist,
                        direction    = signal.direction,
                        setup_label  = setup_lbl,
                        index_prob   = prob,
                        features     = feats,
                    )
                    positions[stk] = pos

        result.trades = all_trades
        result.final_capital = round(cash + sum(p.entry_price * p.qty for p in positions.values()), 2)

        # Update final capital on stock results
        for stk, s_res in result.stock_results.items():
            s_res.final_capital = s_res.initial_capital + sum(t.net_pnl for t in s_res.trades)

        return result
