# ─────────────────────────────────────────────────────────────────
#  backtest/engines/master_to_stock.py
#  Master Spot Index Signal → Stock Cash Delivery Execution Engine.
#
#  Mathematical Basis:
#    - Individual stock charts on 1h/15m suffer from indicator noise
#      and penny wiggles, triggering false breakouts and STT tax bleed.
#    - The Master Spot Index (NIFTY50 / BANKNIFTY) reflects true
#      institutional macro order flow cleanly without individual share noise.
#
#  Execution Architecture:
#    - Signals generated strictly from clean Spot Index (NIFTY50 or BANKNIFTY)
#    - Trades executed on individual stock shares (RELIANCE.NS, HDFCBANK.NS, etc.)
#    - Sizing: Pure zero-leverage CNC Delivery (trade_type="CNC")
#    - SEBI Compliance: Short signals hard-blocked in CNC Cash Delivery
#    - Costs: Zero Brokerage (Zerodha/Dhan model; real STT/Exchange tax applied)
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from loguru import logger

from config.india_settings import INITIAL_CAPITAL_INR, INDIA_MAX_RISK_PER_TRADE_PCT, INDIA_SIGNAL_THRESHOLD
from data.india.nse_client import NSEClient
from indicators import add_all_indicators
from probability.unified_cross_scorer import UnifiedCrossScorer
from engine.india_costs import calculate_round_trip_cost


@dataclass
class MasterStockTrade:
    symbol:        str
    master_index:  str
    entry_time:    datetime
    exit_time:     datetime
    direction:     str
    entry_price:   float
    exit_price:    float
    qty:           float
    gross_pnl:     float
    cost_inr:      float
    net_pnl:       float
    net_pnl_pct:   float
    exit_type:     str
    setup_label:   str
    index_prob:    float
    candles_held:  int = 0
    features:      dict[str, float] = field(default_factory=dict)


@dataclass
class MasterStockResult:
    symbol:          str
    master_index:    str
    interval:        str
    period_start:    datetime
    period_end:      datetime
    candles:         int
    initial_capital: float
    trades:          list[MasterStockTrade] = field(default_factory=list)
    final_capital:   float = 0.0

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
    def avg_win(self) -> float:
        w = [t.net_pnl for t in self.winning_trades]
        return sum(w) / len(w) if w else 0.0

    @property
    def avg_loss(self) -> float:
        l = [t.net_pnl for t in self.losing_trades]
        return sum(l) / len(l) if l else 0.0

    @property
    def profit_factor(self) -> float:
        gp = sum(t.net_pnl for t in self.winning_trades)
        gl = abs(sum(t.net_pnl for t in self.losing_trades))
        return round(gp / gl, 2) if gl else float("inf")

    @property
    def max_drawdown_pct(self) -> float:
        capital = self.initial_capital
        peak, max_dd = capital, 0.0
        for t in self.trades:
            capital += t.net_pnl
            peak     = max(peak, capital)
            dd       = (peak - capital) / peak * 100
            max_dd   = max(max_dd, dd)
        return round(max_dd, 2)

    @property
    def total_costs_inr(self) -> float: return sum(t.cost_inr for t in self.trades)

    @property
    def avg_candles_held(self) -> float:
        return sum(t.candles_held for t in self.trades) / self.total_trades if self.trades else 0.0


class MasterIndexToStockEngine:
    """
    Simulates: Clean Spot Index signal generation → Stock CNC Delivery execution.
    """

    def __init__(
        self,
        stock_symbol: str = "HDFCBANK.NS",
        master_index: str = "BANKNIFTY",  # or "NIFTY50"
        interval:     str = "1h",
        bars:         int = 1400,
        threshold:    float = INDIA_SIGNAL_THRESHOLD,
        atr_sl_mult:  float = 1.0,
        atr_tp_mult:  float = 2.5,
        capital:      float = INITIAL_CAPITAL_INR,
        vix:          float = 16.0,
        max_timeout:  int   = 0,
        stagger_entries_per_day: int = 0,
    ):
        self.stock_symbol = stock_symbol
        self.master_index = master_index
        self.interval     = interval
        self.bars         = bars
        self.threshold    = threshold
        self.atr_sl_mult  = atr_sl_mult
        self.atr_tp_mult  = atr_tp_mult
        self.capital       = capital
        self.vix           = vix
        self.max_timeout   = max_timeout
        self.stagger_entries_per_day = stagger_entries_per_day
        self.client        = NSEClient()
        self.scorer        = UnifiedCrossScorer(
            symbol      = master_index,
            threshold   = threshold,
            atr_sl_mult = atr_sl_mult,
            atr_tp_mult = atr_tp_mult,
            interval    = interval,
            current_vix = vix,
        )

    def run(self) -> MasterStockResult:
        logger.info(f"Fetching {self.bars} × {self.interval} candles for Master {self.master_index} & Stock {self.stock_symbol}...")
        df_idx = add_all_indicators(self.client.get_ohlcv(self.master_index, self.interval, self.bars))
        df_stk = add_all_indicators(self.client.get_ohlcv(self.stock_symbol, self.interval, self.bars))

        # Align timestamps by intersection
        common_idx = df_idx.index.intersection(df_stk.index)
        df_idx = df_idx.loc[common_idx]
        df_stk = df_stk.loc[common_idx]

        WARMUP = min(200, len(df_idx) // 4)
        result = MasterStockResult(
            symbol          = self.stock_symbol,
            master_index    = self.master_index,
            interval        = self.interval,
            period_start    = df_idx.index[WARMUP],
            period_end      = df_idx.index[-1],
            candles         = len(df_idx) - WARMUP,
            initial_capital = self.capital,
            final_capital   = self.capital,
        )

        cash = self.capital
        consec_losses = 0
        cooldown_until = -1

        i = WARMUP
        while i < len(df_idx) - 1:
            if i < cooldown_until:
                i += 1
                continue

            row_idx   = df_idx.iloc[i]
            slice_idx = df_idx.iloc[: i + 1]

            # 1. Generate signal strictly from clean Master Spot Index
            signal = self.scorer.score(row_idx, slice_idx)

            # In Cash Delivery (CNC), overnight shorting is forbidden by SEBI
            if not signal.is_tradeable() or signal.direction == "SHORT":
                i += 1
                continue

            # 2. Execute on individual Stock price at index i
            row_stk   = df_stk.iloc[i]
            entry_stk = float(row_stk["close"])
            atr_stk   = row_stk.get("atr") if hasattr(row_stk, "get") else row_stk["atr"]
            if math.isnan(atr_stk) or atr_stk <= 0:
                atr_stk = entry_stk * 0.008

            sl_dist_stk = atr_stk * self.atr_sl_mult
            sl_stk      = round(entry_stk - sl_dist_stk, 2)
            tp_stk      = round(entry_stk + sl_dist_stk * self.atr_tp_mult, 2)

            # Cash cap: zero leverage in pure CNC Delivery
            risk_inr = cash * INDIA_MAX_RISK_PER_TRADE_PCT / 100
            qty      = risk_inr / sl_dist_stk if sl_dist_stk > 0 else 0
            if entry_stk > 0:
                qty = min(qty, cash / entry_stk)
            if qty <= 0:
                i += 1
                continue

            trade = self._simulate_forward(df_stk, signal, i, qty, entry_stk, sl_dist_stk, sl_stk, tp_stk)

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

                i += (df_stk.index.get_loc(trade.exit_time) - i) if trade.exit_time in df_stk.index else 1
                continue

            i += 1

        result.final_capital = round(cash, 2)
        return result

    def _simulate_forward(
        self,
        df_stk:    pd.DataFrame,
        signal,
        entry_idx: int,
        qty:       float,
        entry_stk: float,
        sl_dist:   float,
        sl:        float,
        tp:        float,
    ) -> MasterStockTrade | None:
        direction = signal.direction
        timeout   = self.max_timeout if self.max_timeout > 0 else len(df_stk)

        for j in range(entry_idx + 1, min(entry_idx + timeout + 1, len(df_stk))):
            candle       = df_stk.iloc[j]
            high, low    = float(candle["high"]), float(candle["low"])
            candles_held = j - entry_idx

            if direction == "LONG":
                if low <= sl:
                    exit_type = "TRAIL_STOP" if sl >= entry_stk else "STOP_LOSS"
                    return self._make_trade(signal, df_stk, entry_idx, j, entry_stk, sl, exit_type, qty, candles_held)
                if high >= tp:
                    return self._make_trade(signal, df_stk, entry_idx, j, entry_stk, tp, "TAKE_PROFIT", qty, candles_held)

                # Trailing stop
                if high >= entry_stk + sl_dist * 1.5 and sl < entry_stk + sl_dist * 0.5:
                    sl = round(entry_stk + sl_dist * 0.5, 2)
                elif high >= entry_stk + sl_dist * 1.0 and sl < entry_stk:
                    sl = round(entry_stk, 2)

        j = min(entry_idx + timeout, len(df_stk) - 1)
        exit_price = float(df_stk.iloc[j]["close"])
        candles_held = j - entry_idx
        return self._make_trade(signal, df_stk, entry_idx, j, entry_stk, exit_price, "TIMEOUT", qty, candles_held)

    def _make_trade(
        self,
        signal,
        df_stk:      pd.DataFrame,
        entry_idx:   int,
        exit_idx:    int,
        entry_price: float,
        exit_price:  float,
        exit_type:   str,
        qty:         float,
        candles_held: int,
    ) -> MasterStockTrade:
        gross_pnl = (exit_price - entry_price) * qty
        buy_cost, sell_cost = calculate_round_trip_cost(entry_price, exit_price, qty, "CNC")
        cost_inr  = buy_cost.total + sell_cost.total
        net_pnl   = gross_pnl - cost_inr
        net_pct   = net_pnl / (entry_price * qty) * 100 if (entry_price * qty) > 0 else 0.0

        reason_str = signal.reason or ""
        setup_lbl  = "DUAL_CONFIRM" if "DUAL CONFIRM" in reason_str else ("BAND_BOUNCE" if "BAND BOUNCE" in reason_str else "PULLBACK")

        return MasterStockTrade(
            symbol       = self.stock_symbol,
            master_index = self.master_index,
            entry_time   = df_stk.index[entry_idx].to_pydatetime(),
            exit_time    = df_stk.index[exit_idx].to_pydatetime(),
            direction    = signal.direction,
            entry_price  = round(entry_price, 2),
            exit_price   = round(exit_price, 2),
            qty          = qty,
            gross_pnl    = round(gross_pnl, 2),
            cost_inr     = round(cost_inr, 2),
            net_pnl      = round(net_pnl, 2),
            net_pnl_pct  = round(net_pct, 4),
            exit_type    = exit_type,
            setup_label  = setup_lbl,
            index_prob   = signal.probability,
        )
