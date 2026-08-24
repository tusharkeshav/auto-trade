# ─────────────────────────────────────────────────────────────────
#  demo_trade.py
#  Places one forced paper LONG and watches it live.
#  Uses tight SL/TP so the lifecycle completes quickly.
#
#  Usage:
#      source .venv/bin/activate && python demo_trade.py
# ─────────────────────────────────────────────────────────────────

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime           import datetime, timezone
from loguru             import logger
from rich.console       import Console
from rich.table         import Table
from rich               import box

from data.binance_client       import BinanceClient
from engine                    import Portfolio, OrderManager, Ledger
from probability.signal_scorer import TradeSignal

console = Console()
SYMBOL  = "BTCUSDT"
CHECK_INTERVAL = 10   # seconds between price checks in demo


def section(title: str):
    console.print(f"\n[bold blue]{'═' * 60}[/]")
    console.print(f"[bold cyan]  {title}[/]")
    console.print(f"[bold blue]{'═' * 60}[/]")


def build_demo_signal(symbol: str, price: float, atr: float) -> TradeSignal:
    """
    Craft a demo LONG signal with TIGHT SL/TP so the lifecycle
    resolves quickly (within a few minutes at normal volatility).

    SL  = 0.15% below entry  (tight — may hit on normal wick)
    TP1 = 0.15% above entry  (1:1 — books 50% very fast)
    TP2 = 0.22% above entry  (1.5:1 — trails the rest)
    """
    sl_pct  = 0.0015   # 0.15%
    tp1_pct = 0.0015
    tp2_pct = 0.0022

    sl  = round(price * (1 - sl_pct), 2)
    tp1 = round(price * (1 + tp1_pct), 2)
    tp2 = round(price * (1 + tp2_pct), 2)

    return TradeSignal(
        symbol       = symbol,
        timestamp    = datetime.utcnow(),
        direction    = "LONG",
        probability  = 72.0,   # demo — pretend we have a signal
        confidence   = "MEDIUM",
        raw_score    = 44.0,
        entry_price  = price,
        stop_loss    = sl,
        take_profit1 = tp1,
        take_profit2 = tp2,
        risk_amount  = round(price - sl, 2),
        breakdown    = [],
        reason       = "DEMO TRADE — forced for testing purposes",
    )


def print_position_status(pos, current_price: float, trades):
    color  = "green" if pos.unrealized_pnl(current_price) >= 0 else "red"
    sign   = "+" if pos.unrealized_pnl(current_price) >= 0 else ""
    pnl    = pos.unrealized_pnl(current_price)
    pnl_pc = pos.unrealized_pnl_pct(current_price)

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column(style="dim", width=20)
    t.add_column(style="bold", width=20)
    t.add_column(style="dim", width=20)
    t.add_column(style="bold", width=20)

    status = "🔄 OPEN"
    sl_label = f"[red]${pos.stop_loss:,.2f}[/] ← stop-hunt-safe"

    t.add_row("Entry",       f"[cyan]${pos.entry_price:,.2f}[/]",  "Current",    f"[bold cyan]${current_price:,.2f}[/]")
    t.add_row("Stop Loss",   sl_label,                              "TP1",        f"${ pos.take_profit1:,.2f}")
    t.add_row("TP2 Target",  f"${pos.take_profit2:,.2f}",          "Size",       f"{pos.size_remaining:.6f} BTC")
    t.add_row("Unr. P&L",   f"[{color}]{sign}${pnl:.4f} ({sign}{pnl_pc:.3f}%)[/]",
              "Status",      status)

    console.print(t)
    if trades:
        console.print(f"  [dim]Completed trade records: {len(trades)}[/]")


def main():
    section("🤖 DEMO TRADE — AutoTrader Paper Bot")
    console.print(
        "\n  This demo places a forced LONG with tight SL/TP so you can\n"
        "  watch the full trade lifecycle: open → TP1 hit → breakeven SL\n"
        "  → TP2 hit (or SL). All in real paper money.\n"
    )

    client    = BinanceClient()
    portfolio = Portfolio()
    ledger    = Ledger()
    orders    = OrderManager(portfolio, ledger)

    # ── Fetch current price and ATR ───────────────────────────────
    section("📡 Fetching Live Data")
    from indicators import add_all_indicators
    df     = client.get_ohlcv(SYMBOL, "1h", 50)
    df     = add_all_indicators(df)
    price  = client.get_price(SYMBOL)
    atr    = df["atr"].iloc[-1]

    console.print(f"  [bold white]{SYMBOL}[/]  Current Price: [bold cyan]${price:,.2f}[/]   ATR: [yellow]${atr:,.2f}[/]")

    # ── Build and place demo signal ───────────────────────────────
    section("📈 Opening Demo LONG Position")
    signal   = build_demo_signal(SYMBOL, price, atr)
    position = orders.try_open(signal)

    if not position:
        console.print("[red]Failed to open position — check portfolio capital.[/]")
        return

    console.print(f"\n  [bold green]✅ Position opened![/]")
    console.print(f"  Entry  : [cyan]${position.entry_price:,.2f}[/]")
    console.print(f"  SL     : [red]${position.stop_loss:,.2f}[/]  ({((position.entry_price - position.stop_loss)/position.entry_price*100):.3f}% below)")
    console.print(f"  TP1    : [green]${position.take_profit1:,.2f}[/]  → book 50% here")
    console.print(f"  TP2    : [green]${position.take_profit2:,.2f}[/]  → trail rest")
    console.print(f"  Size   : [yellow]{position.size_total:.6f} BTC[/]  (2% risk of ${portfolio.initial_capital:,.0f})")

    # ── Live monitoring loop ──────────────────────────────────────
    section("👁️  Live Price Monitor (checking every 10s)")
    console.print("  [dim]Watch the position update in real time. Press Ctrl+C to stop.[/]\n")

    start = datetime.now(timezone.utc)
    try:
        check_num = 0
        while portfolio.positions:   # loop until position closed
            check_num += 1
            current_price = client.get_price(SYMBOL)

            elapsed = int((datetime.now(timezone.utc) - start).total_seconds())
            console.print(
                f"  [dim]#{check_num:>3}  {elapsed:>4}s  [/]"
                f"Price: [bold cyan]${current_price:,.2f}[/]  │  "
                f"SL: [red]${position.stop_loss:,.2f}[/]  │  "
                f"TP1: [green]${position.take_profit1:,.2f}[/]  │  "
                f"TP2: [green]${position.take_profit2:,.2f}[/]"
            )
            print_position_status(position, current_price, ledger.recent())

            # Check SL / TP
            orders.check_positions({SYMBOL: current_price})

            if position.id not in portfolio.positions:
                console.print("\n  [bold yellow]Position closed![/]")
                break

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        console.print("\n  [yellow]Demo stopped by user.[/]")

    # ── Final summary ─────────────────────────────────────────────
    section("📊 Demo Results")
    total_pnl = portfolio.total_realized_pnl
    sign      = "+" if total_pnl >= 0 else ""

    console.print(f"  Trades completed : {len(ledger._records)}")
    console.print(f"  Realized P&L     : [{('green' if total_pnl >= 0 else 'red')}]{sign}${total_pnl:.4f}[/]")
    console.print(f"  Win Rate         : {portfolio.win_rate():.1f}%")

    for r in ledger.recent():
        icon = "✅" if r.is_win else "❌"
        console.print(
            f"\n  {icon} {r.close_type.replace('_',' '):<18}  "
            f"Entry: ${r.entry_price:,.2f}  →  Exit: ${r.exit_price:,.2f}  │  "
            f"P&L: [{('green' if r.pnl >= 0 else 'red')}]{('+' if r.pnl >= 0 else '')}${r.pnl:.4f}[/]"
        )
    console.print()


if __name__ == "__main__":
    main()
