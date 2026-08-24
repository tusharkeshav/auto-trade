# ─────────────────────────────────────────────────────────────────
#  backtest/engines/expiry_pin.py
#  Walk-forward backtest for options expiry pinning strategy.
#
#  Data: NIFTY/BANKNIFTY daily OHLCV from yfinance (free, unlimited)
#  Signal: ExpiryPinScorer — trades toward max pain on Wed/Thu
#  Max pain: computed from historical options chain (via cached snapshots)
#
#  NOTE: NSE options chain is live-only (no historical API available free).
#        Backtest simulates the strategy using:
#          1. Actual daily OHLCV (yfinance)
#          2. Reconstructed max pain from first principles:
#             - Backtests use fixed max_pain_offset_pct per week cycle
#             - Real forward-test uses live NSE chain data
#
#  For genuine historical backtesting, max pain must come from
#  archived options chain data (NSEPython, Sensibull exports, etc.)
#  This engine uses a synthetic max pain model for simulation:
#    max_pain = rolling 5-day EMA of close (proxy: price tends to
#    close near recent average, which approximates max pain gravity)
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
from loguru import logger

from config.india_settings import INITIAL_CAPITAL_INR, INDIA_ATR_SL_MULT, INDIA_ATR_TP_MULT
from data.india.nse_client import NSEClient
from engine.india_costs import effective_cost_pct
from indicators import add_all_indicators

IST = ZoneInfo("Asia/Kolkata")

# Expiry pin specific constants
_EXPIRY_DAY     = 3      # Thursday weekday index
_WED_ENTRY_DAY  = 2      # Wednesday
_MIN_DIST_PCT   = 0.3    # ignore signals within 0.3% of synthetic max pain
_MAX_HOLD_DAYS  = 2      # hold at most 2 daily bars (Wed entry → Thu close)


@dataclass
class ExpiryPinTrade:
    symbol:        str
    direction:     str
    entry_time:    datetime
    exit_time:     datetime
    entry_price:   float
    exit_price:    float
    qty:           float
    pnl:           float
    pnl_pct:       float
    exit_type:     str
    probability:   float
    candles_held:  int
    cost_inr:      float
    dist_pct:      float   # spot vs synthetic max pain at entry
    synthetic_mp:  float   # synthetic max pain used


@dataclass
class ExpiryPinResult:
    symbol:          str
    interval:        str
    period_start:    datetime
    period_end:      datetime
    candles:         int
    initial_capital: float
    trades:          list[ExpiryPinTrade] = field(default_factory=list)
    final_capital:   float = 0.0

    @property
    def total_trades(self) -> int: return len(self.trades)

    @property
    def winning_trades(self) -> list: return [t for t in self.trades if t.pnl > 0]

    @property
    def losing_trades(self)  -> list: return [t for t in self.trades if t.pnl <= 0]

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
    def monthly_returns(self) -> dict:
        from collections import defaultdict
        capital, month_start, monthly_pnl = self.initial_capital, {}, defaultdict(float)
        for t in self.trades:
            mk = t.exit_time.strftime("%Y-%m")
            if mk not in month_start: month_start[mk] = capital
            monthly_pnl[mk] += t.pnl
            capital          += t.pnl
        return {mk: pnl / month_start[mk] * 100 for mk, pnl in monthly_pnl.items()}


class ExpiryPinEngine:
    """
    Walk-forward backtest for options expiry pinning.

    Uses daily OHLCV bars. Enters on Wednesday close or Thursday open,
    exits on Thursday close or SL/TP hit.

    Synthetic max pain = 5-day EMA of close (proxy for where price
    attracts on expiry — real backtesting needs archived options data).
    """

    def __init__(
        self,
        symbol:        str   = "NIFTY50",
        bars:          int   = 500,
        capital:       float = INITIAL_CAPITAL_INR,
        atr_sl_mult:   float = INDIA_ATR_SL_MULT,
        atr_tp_mult:   float = INDIA_ATR_TP_MULT,
        risk_pct:      float = 1.0,
        min_dist_pct:  float = _MIN_DIST_PCT,
    ):
        self.symbol       = symbol
        self.bars         = bars
        self.capital      = capital
        self.atr_sl_mult  = atr_sl_mult
        self.atr_tp_mult  = atr_tp_mult
        self.risk_pct     = risk_pct
        self.min_dist_pct = min_dist_pct
        self.client       = NSEClient()

    def _load_data(self) -> pd.DataFrame:
        logger.info(f"Fetching {self.bars} daily candles for {self.symbol}...")
        df = self.client.get_ohlcv(self.symbol, "1d", self.bars)
        df = add_all_indicators(df)
        # Synthetic max pain: 5-day EMA of close
        df["synthetic_mp"] = df["close"].ewm(span=5, adjust=False).mean()
        df["mp_dist_pct"]  = (df["close"] - df["synthetic_mp"]) / df["synthetic_mp"] * 100
        return df

    def run(self, external_df: pd.DataFrame | None = None) -> ExpiryPinResult:
        df = external_df if external_df is not None else self._load_data()
        if "synthetic_mp" not in df.columns:
            df["synthetic_mp"] = df["close"].ewm(span=5, adjust=False).mean()
            df["mp_dist_pct"]  = (df["close"] - df["synthetic_mp"]) / df["synthetic_mp"] * 100

        WARMUP = 20   # need 5-day EMA to stabilise
        result = ExpiryPinResult(
            symbol          = self.symbol,
            interval        = "1d",
            period_start    = df.index[WARMUP],
            period_end      = df.index[-1],
            candles         = len(df) - WARMUP,
            initial_capital = self.capital,
            final_capital   = self.capital,
        )
        cash = self.capital
        skip_until = -1

        for i in range(WARMUP, len(df) - 1):
            if i < skip_until:
                continue

            ts       = df.index[i]
            weekday  = ts.weekday()

            # Only trade on Wednesday (enter next day Thu) or Thursday
            if weekday not in (_WED_ENTRY_DAY, _EXPIRY_DAY):
                continue

            row        = df.iloc[i]
            dist_pct   = float(row["mp_dist_pct"])
            synth_mp   = float(row["synthetic_mp"])
            close      = float(row["close"])
            atr_val    = float(row["atr"]) if not math.isnan(row["atr"]) else close * 0.008

            # Noise filter
            if abs(dist_pct) < self.min_dist_pct:
                continue

            direction = "SHORT" if dist_pct > 0 else "LONG"

            # Probability score based on distance (mirrors ExpiryPinScorer)
            dist_abs  = abs(dist_pct)
            base_prob = 60.0 + min(25.0, (dist_abs - self.min_dist_pct) / (0.8 - self.min_dist_pct) * 25.0)
            probability = min(92.0, base_prob)

            # Entry = next bar open
            entry_idx   = i + 1
            entry_price = float(df.iloc[entry_idx]["open"])

            # SL/TP
            sl_dist = atr_val * self.atr_sl_mult
            if direction == "LONG":
                sl = round(entry_price - sl_dist, 2)
                tp = round(entry_price + sl_dist * self.atr_tp_mult, 2)
            else:
                sl = round(entry_price + sl_dist, 2)
                tp = round(entry_price - sl_dist * self.atr_tp_mult, 2)

            # Sizing
            risk_inr = cash * self.risk_pct / 100
            qty      = risk_inr / sl_dist if sl_dist > 0 else 0
            if qty <= 0:
                continue

            # Simulate forward: max hold = _MAX_HOLD_DAYS bars
            end_idx = min(entry_idx + _MAX_HOLD_DAYS, len(df) - 1)
            exit_price, exit_type, exit_idx = _simulate_forward(
                df, direction, entry_idx, end_idx, sl, tp
            )

            candles_held = exit_idx - entry_idx
            if direction == "LONG":
                gross_pnl = (exit_price - entry_price) * qty
            else:
                gross_pnl = (entry_price - exit_price) * qty

            cost_pct = effective_cost_pct(entry_price, exit_price, qty, "MIS") / 100
            cost_inr = (entry_price * qty + exit_price * qty) * cost_pct
            net_pnl  = gross_pnl - cost_inr
            pnl_pct  = net_pnl / (entry_price * qty) * 100

            trade = ExpiryPinTrade(
                symbol       = self.symbol,
                direction    = direction,
                entry_time   = df.index[entry_idx].to_pydatetime(),
                exit_time    = df.index[exit_idx].to_pydatetime(),
                entry_price  = entry_price,
                exit_price   = round(exit_price, 2),
                qty          = qty,
                pnl          = round(net_pnl, 2),
                pnl_pct      = round(pnl_pct, 4),
                exit_type    = exit_type,
                probability  = round(probability, 1),
                candles_held = candles_held,
                cost_inr     = round(cost_inr, 2),
                dist_pct     = round(dist_pct, 2),
                synthetic_mp = round(synth_mp, 2),
            )

            cash += net_pnl
            result.trades.append(trade)
            skip_until = exit_idx + 1

        result.final_capital = round(cash, 2)
        logger.success(
            f"Expiry pin backtest done: {result.total_trades} trades, "
            f"P&L ₹{result.total_pnl:+,.2f} ({result.total_pnl_pct:+.2f}%)"
        )
        return result

    def print_report(self, r: ExpiryPinResult) -> None:
        from rich.console import Console
        from rich.table   import Table
        from rich         import box

        c = Console()

        def sep():
            c.print("[bold magenta]" + "═" * 68 + "[/]")

        sign    = "+" if r.total_pnl >= 0 else ""
        pnl_col = "green" if r.total_pnl >= 0 else "red"

        sep()
        c.print(f"[bold cyan]  EXPIRY PIN BACKTEST — {r.symbol}  (Daily)[/]")
        c.print("[dim]  Strategy: spot vs synthetic max pain (5-day EMA), trade Wed/Thu[/]")
        c.print("[yellow]  NOTE: Synthetic max pain is a proxy. Real results need archived chain data.[/]")
        sep()
        c.print(f"  [dim]Period   :[/]  {r.period_start.strftime('%Y-%m-%d')}  →  {r.period_end.strftime('%Y-%m-%d')}")
        c.print(f"  [dim]Candles  :[/]  {r.candles}  daily bars")

        c.print(f"\n  [bold]── Performance ─────────────────────────────────────[/]")
        c.print(f"  Initial Capital  : [cyan]₹{r.initial_capital:>12,.2f}[/]")
        c.print(f"  Final Capital    : [cyan]₹{r.final_capital:>12,.2f}[/]")
        c.print(f"  Total P&L        : [{pnl_col}]{sign}₹{r.total_pnl:>10,.2f}  ({sign}{r.total_pnl_pct:.2f}%)[/]")
        c.print(f"  Max Drawdown     : [red]{r.max_drawdown_pct:.2f}%[/]")

        c.print(f"\n  [bold]── Trade Statistics ─────────────────────────────────[/]")
        c.print(f"  Total Trades     : [bold]{r.total_trades}[/]")
        c.print(f"  Win Rate         : [yellow]{r.win_rate:.1f}%[/]  ({len(r.winning_trades)}W / {len(r.losing_trades)}L)")
        c.print(f"  Profit Factor    : [{'green' if r.profit_factor >= 1.5 else 'yellow'}]{r.profit_factor:.2f}[/]  [dim](≥1.5 good)[/]")
        c.print(f"  Avg Win          : [green]+₹{r.avg_win:,.2f}[/]")
        c.print(f"  Avg Loss         : [red]₹{r.avg_loss:,.2f}[/]")

        c.print(f"\n  [bold]── Monthly Returns ──────────────────────────────────[/]")
        for month, pct in sorted(r.monthly_returns.items()):
            col  = "green" if pct >= 0 else "red"
            sign = "+" if pct >= 0 else ""
            c.print(f"  {month} : [{col}]{sign}{pct:.2f}%[/]")

        c.print(f"\n  [bold]── Trade Log ────────────────────────────────────────[/]")
        tbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
        tbl.add_column("Entry",      width=12)
        tbl.add_column("Dir",        width=6)
        tbl.add_column("Entry ₹",    justify="right", width=10)
        tbl.add_column("Exit ₹",     justify="right", width=10)
        tbl.add_column("Dist%",      justify="right", width=7)
        tbl.add_column("Exit",       width=12)
        tbl.add_column("P&L",        justify="right", width=12)

        for t in r.trades:
            pc  = "green" if t.pnl >= 0 else "red"
            sgn = "+" if t.pnl >= 0 else ""
            dc  = "green" if t.direction == "LONG" else "red"
            tbl.add_row(
                t.entry_time.strftime("%Y-%m-%d"),
                f"[{dc}]{t.direction}[/]",
                f"₹{t.entry_price:>8,.1f}",
                f"₹{t.exit_price:>8,.1f}",
                f"{t.dist_pct:+.2f}%",
                t.exit_type,
                f"[{pc}]{sgn}₹{t.pnl:>8,.2f}[/]",
            )
        c.print(tbl)
        sep()


def _simulate_forward(
    df:        pd.DataFrame,
    direction: str,
    entry_idx: int,
    end_idx:   int,
    sl:        float,
    tp:        float,
) -> tuple[float, str, int]:
    """
    Walk forward bar by bar. Returns (exit_price, exit_type, exit_idx).
    TIMEOUT at end_idx close if SL/TP not hit.
    """
    for j in range(entry_idx + 1, end_idx + 1):
        high = float(df.iloc[j]["high"])
        low  = float(df.iloc[j]["low"])

        if direction == "LONG":
            if low  <= sl: return sl, "STOP_LOSS",   j
            if high >= tp: return tp, "TAKE_PROFIT", j
        else:
            if high >= sl: return sl, "STOP_LOSS",   j
            if low  <= tp: return tp, "TAKE_PROFIT", j

    exit_price = float(df.iloc[end_idx]["close"])
    return exit_price, "TIMEOUT", end_idx
