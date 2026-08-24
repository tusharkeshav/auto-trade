# ─────────────────────────────────────────────────────────────────
#  main.py  —  AutoTrader Paper Trading Bot
#
#  Usage:
#      source .venv/bin/activate && python main.py
#
#  Web Dashboard:
#      http://localhost:8080  (auto-refreshes every 5s)
#
#  Stop:
#      Ctrl+C  →  saves ledger to Parquet and exits cleanly
#
#  Loop behaviour (every PRICE_CHECK_INTERVAL seconds):
#    1. Fetch live prices for all symbols (fast REST call)
#    2. Check open positions against SL / TP1 / TP2
#    3. Every SIGNAL_SCAN_INTERVAL: fetch candles + compute indicators
#       + score probability + try to open new positions
#    4. Refresh the live dashboard
# ─────────────────────────────────────────────────────────────────

import sys
import os
import time
from datetime         import datetime, timezone
from loguru           import logger
from rich.live        import Live
from rich.console     import Console

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings              import (
    SYMBOLS, DEFAULT_INTERVAL, DEFAULT_LIMIT,
    SIGNAL_THRESHOLD, MACRO_SESSION_START, MACRO_SESSION_END,
    MOMENTUM_THRESHOLD, MOMENTUM_ADX_MIN,
    MOMENTUM_ATR_SL_MULT, MOMENTUM_ATR_TP_RATIO,
    MOMENTUM_INTERVAL, MOMENTUM_LIMIT, MOMENTUM_SYMBOLS,
)
from data.binance_client           import BinanceClient
from indicators                    import add_all_indicators
from probability                   import SignalScorer
from probability.momentum_scorer   import MomentumScorer
from engine                        import Portfolio, OrderManager, Ledger
from engine.regime                 import detect_regime
from dashboard                     import build_layout
from dashboard.web_server          import BotState, start_web_server

# ─────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────

PRICE_CHECK_INTERVAL  = 30     # seconds between price checks + SL/TP monitoring
CANDLE_INTERVAL_MIN   = 15     # primary timeframe: 15m candles
MOMENTUM_INTERVAL_MIN = 240    # momentum timeframe: 4h candles (BTC only)
WEB_PORT              = 8080   # browser dashboard port

console = Console()


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

def fetch_prices(client: BinanceClient, symbols: list[str]) -> dict[str, float]:
    """Fetch live prices for all symbols in one pass."""
    prices = {}
    for symbol in symbols:
        try:
            prices[symbol] = client.get_price(symbol)
        except Exception as e:
            logger.warning(f"Price fetch failed for {symbol}: {e}")
    return prices


def secs_until_next_candle_close(interval_min: int = 15) -> float:
    """
    Returns the number of seconds until the next 15m candle closes.
    Candles close at :00, :15, :30, :45 past each hour UTC.
    We add a 5-second buffer so the close is complete on Binance's side.
    """
    now = datetime.now(timezone.utc)
    seconds_since_hour = now.minute * 60 + now.second
    interval_sec = interval_min * 60
    elapsed_in_candle = seconds_since_hour % interval_sec
    remaining = interval_sec - elapsed_in_candle
    return remaining + 5.0   # 5s buffer


def scan_signals(
    client:          BinanceClient,
    mr_scorer:       SignalScorer,
    momentum_scorer: MomentumScorer,
    symbols:         list[str],
) -> tuple[dict[str, object], dict[str, str]]:
    """
    For each symbol: fetch candles, compute indicators, detect market
    regime (mean-reversion vs momentum), then score with the appropriate
    scorer.

    Returns (signals dict, regime dict).
    """
    signals = {}
    regimes = {}
    for symbol in symbols:
        try:
            df = client.get_ohlcv(symbol, DEFAULT_INTERVAL, DEFAULT_LIMIT)
            df = add_all_indicators(df)

            last = df.iloc[-2]   # last fully closed candle

            # Detect regime
            decision = detect_regime(last)
            regimes[symbol] = decision.regime
            logger.info(f"[{symbol}] Regime: {decision.regime} ({decision.reason})")

            # Score with appropriate strategy
            if decision.regime == "MOMENTUM":
                momentum_scorer.symbol = symbol
                signals[symbol] = momentum_scorer.score(last)
            else:
                mr_scorer.symbol = symbol
                signals[symbol] = mr_scorer.score(last)

        except Exception as e:
            logger.warning(f"Signal scan failed for {symbol}: {e}")
            signals[symbol] = None
            regimes[symbol] = "ERROR"
    return signals, regimes


def scan_momentum_4h(
    client:          BinanceClient,
    momentum_scorer: MomentumScorer,
) -> dict[str, object]:
    """
    4h momentum scan — BTC only.
    Fetches 4h candles, computes indicators, scores with momentum scorer.

    Only called when a 4h candle closes (:00 past every 4th hour UTC).
    Position guard in OrderManager prevents double-entry.
    """
    signals = {}
    for symbol in MOMENTUM_SYMBOLS:
        try:
            df = client.get_ohlcv(symbol, MOMENTUM_INTERVAL, MOMENTUM_LIMIT)
            df = add_all_indicators(df)
            momentum_scorer.symbol = symbol
            last = df.iloc[-2]   # last fully closed 4h candle
            signals[symbol] = momentum_scorer.score(last)
            if signals[symbol].is_tradeable():
                logger.info(f"[{symbol}] 4h momentum signal: {signals[symbol]}")
        except Exception as e:
            logger.warning(f"4h momentum scan failed for {symbol}: {e}")
            signals[symbol] = None
    return signals


# ─────────────────────────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Bootstrap ─────────────────────────────────────────────────
    console.print("\n[bold cyan]🤖  AutoTrader Paper Bot  —  Starting up...[/]\n")

    client          = BinanceClient()
    portfolio       = Portfolio()
    ledger          = Ledger()
    orders          = OrderManager(portfolio, ledger)
    mr_scorer       = SignalScorer(long_threshold=SIGNAL_THRESHOLD)
    momentum_scorer = MomentumScorer(
        symbol="BTCUSDT",
        long_threshold=MOMENTUM_THRESHOLD,
        momentum_adx_min=MOMENTUM_ADX_MIN,
        atr_sl_mult=MOMENTUM_ATR_SL_MULT,
        atr_tp_ratio=MOMENTUM_ATR_TP_RATIO,
    )
    state     = BotState()

    # ── Start web dashboard ────────────────────────────────────────
    start_web_server(state, port=WEB_PORT)
    console.print(f"[bold green]🌐  Web dashboard  →  http://localhost:{WEB_PORT}[/]\n")

    start_time         = datetime.now(timezone.utc)
    last_signal_time   = None
    last_momentum_signal = None   # timestamp of last 4h momentum scan

    # ── Wait for first candle boundary before entering loop ───────
    wait = secs_until_next_candle_close(CANDLE_INTERVAL_MIN)
    console.print(f"[dim]⏳  Waiting {wait:.0f}s for next 15m candle close before first scan...[/]")
    time.sleep(wait)

    # Initial state
    prices      = fetch_prices(client, SYMBOLS)
    prev_prices = dict(prices)
    signals, regimes = scan_signals(client, mr_scorer, momentum_scorer, SYMBOLS)
    last_signal_time = datetime.now(timezone.utc).strftime("%H:%M:%S")

    # ── Main loop with live dashboard ─────────────────────────────
    layout = build_layout(
        portfolio, prices, prev_prices, signals,
        ledger, start_time, PRICE_CHECK_INTERVAL,
    )

    try:
        with Live(layout, console=console, refresh_per_second=2, screen=True) as live:
            while True:
                loop_start = time.time()

                # ── 1. Refresh prices ──────────────────────────────
                prev_prices = dict(prices)
                prices      = fetch_prices(client, SYMBOLS)

                # ── 2. Monitor open positions (SL / TP) ───────────
                orders.check_positions(prices)

                # ── 3a. 15m signal scan (BTC + ETH) ───────────────────────
                remaining_15m = secs_until_next_candle_close(CANDLE_INTERVAL_MIN)
                candle_15m_closed = remaining_15m > (CANDLE_INTERVAL_MIN * 60 - PRICE_CHECK_INTERVAL - 10)

                if candle_15m_closed:
                    signals, regimes = scan_signals(client, mr_scorer, momentum_scorer, SYMBOLS)
                    last_signal_time = datetime.now(timezone.utc).strftime("%H:%M:%S")

                    for symbol, signal in signals.items():
                        if (
                            signal
                            and signal.is_tradeable()
                            and signal.probability >= SIGNAL_THRESHOLD
                        ):
                            orders.try_open(signal)

                # ── 3b. 4h momentum scan (BTC only) ─────────────────────
                # Fires every 4 hours when the 4h candle closes.
                # Position guard in OrderManager prevents double-entry
                # if BTC already has a 15m MR position open.
                remaining_4h = secs_until_next_candle_close(MOMENTUM_INTERVAL_MIN)
                candle_4h_closed = remaining_4h > (MOMENTUM_INTERVAL_MIN * 60 - PRICE_CHECK_INTERVAL - 10)

                if candle_4h_closed:
                    signals_4h = scan_momentum_4h(client, momentum_scorer)
                    last_momentum_signal = datetime.now(timezone.utc).strftime("%H:%M:%S")
                    logger.info(f"4h momentum scan fired at {last_momentum_signal}")

                    for symbol, signal in signals_4h.items():
                        if (
                            signal
                            and signal.is_tradeable()
                            and signal.probability >= MOMENTUM_THRESHOLD
                        ):
                            orders.try_open(signal)

                # ── 4. Update shared web state ─────────────────────
                state.update(prices, prev_prices, signals, portfolio, ledger)

                # ── 5. Refresh terminal dashboard ──────────────────
                next_candle_in = int(secs_until_next_candle_close(CANDLE_INTERVAL_MIN))
                layout = build_layout(
                    portfolio, prices, prev_prices, signals,
                    ledger, start_time, next_candle_in, last_signal_time,
                )
                live.update(layout)

                # ── 6. Sleep until next price check ────────────────
                sleep_time = max(0, PRICE_CHECK_INTERVAL - (time.time() - loop_start))
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass

    # ── Graceful shutdown ─────────────────────────────────────────
    console.print("\n[bold yellow]⏹  Stopping bot...[/]")

    if ledger._records:
        path = ledger.save()
        console.print(f"[green]✅  Ledger saved → {path}[/]")

    final_value = portfolio.total_value(prices)
    final_pnl   = portfolio.total_pnl(prices)
    sign        = "+" if final_pnl >= 0 else ""
    console.print(
        f"\n[bold cyan]📊 Final Stats[/]\n"
        f"  Portfolio Value : [cyan]${final_value:,.2f}[/]\n"
        f"  Total P&L       : [{'green' if final_pnl >= 0 else 'red'}]{sign}${final_pnl:,.2f}[/]\n"
        f"  Total Trades    : {portfolio.total_trades}\n"
        f"  Win Rate        : {portfolio.win_rate():.1f}%\n"
    )


if __name__ == "__main__":
    main()
