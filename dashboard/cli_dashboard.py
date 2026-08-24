# ─────────────────────────────────────────────────────────────────
#  dashboard/cli_dashboard.py
#  Beautiful live terminal dashboard using Rich.
#
#  Layout:
#    ┌─ Header: bot name, portfolio value, P&L, time ───────────┐
#    ├─ Portfolio ─┬─ Live Prices ──┬─ Signal Status ───────────┤
#    ├─ Open Positions Table ────────────────────────────────────┤
#    ├─ Recent Trades Table ─────────────────────────────────────┤
#    └─ Status bar: next check countdown ──────────────────────── ┘
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

from datetime import datetime, timezone
from typing   import Optional

from rich                import box
from rich.align          import Align
from rich.columns        import Columns
from rich.console        import Console
from rich.layout         import Layout
from rich.live           import Live
from rich.panel          import Panel
from rich.table          import Table
from rich.text           import Text

from engine.portfolio import Portfolio
from engine.ledger    import Ledger


console = Console()


# ─────────────────────────────────────────────────────────────────
#  Individual panel renderers
# ─────────────────────────────────────────────────────────────────

def _header(portfolio: Portfolio, prices: dict[str, float], start_time: datetime) -> Panel:
    total    = portfolio.total_value(prices)
    pnl      = portfolio.total_pnl(prices)
    pnl_pct  = portfolio.total_pnl_pct(prices)
    pnl_color = "green" if pnl >= 0 else "red"
    pnl_sign  = "+" if pnl >= 0 else ""

    elapsed  = datetime.now(timezone.utc) - start_time
    hours, rem = divmod(int(elapsed.total_seconds()), 3600)
    minutes, seconds = divmod(rem, 60)
    runtime  = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    now_str  = datetime.now(timezone.utc).strftime("%a %d %b %Y  %H:%M:%S UTC")

    text = Text(justify="center")
    text.append("🤖  AutoTrader  ", style="bold white")
    text.append("│  ", style="dim")
    text.append("PAPER TRADING  ", style="bold yellow")
    text.append("│  ", style="dim")
    text.append(f"💰 ${total:>12,.2f}  ", style="bold cyan")
    text.append(f"({pnl_sign}${pnl:,.2f}  {pnl_sign}{pnl_pct:.2f}%)  ", style=f"bold {pnl_color}")
    text.append("│  ", style="dim")
    text.append(f"⏱  {runtime}  ", style="bold white")
    text.append("│  ", style="dim")
    text.append(now_str, style="dim white")

    return Panel(Align.center(text), style="bold blue", height=3)


def _portfolio_panel(portfolio: Portfolio, prices: dict[str, float]) -> Panel:
    total    = portfolio.total_value(prices)
    pnl      = portfolio.total_pnl(prices)
    pnl_pct  = portfolio.total_pnl_pct(prices)
    pnl_color = "green" if pnl >= 0 else "red"
    sign      = "+" if pnl >= 0 else ""

    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim")
    t.add_column(style="bold")

    t.add_row("Total Value",   f"[cyan]${total:>12,.2f}[/]")
    t.add_row("Cash",          f"[white]${portfolio.cash:>12,.2f}[/]")
    t.add_row("P&L",           f"[{pnl_color}]{sign}${pnl:>10,.2f}  ({sign}{pnl_pct:.2f}%)[/]")
    t.add_row("Win Rate",      f"[yellow]{portfolio.win_rate():.1f}%  ({portfolio.winning_trades}W / {portfolio.total_trades - portfolio.winning_trades}L)[/]")
    t.add_row("Open Trades",   f"[bold]{portfolio.open_position_count()}[/]")
    t.add_row("Total Trades",  f"[bold]{portfolio.total_trades}[/]")

    return Panel(t, title="[bold cyan]Portfolio[/]", border_style="cyan", height=10)


def _prices_panel(prices: dict[str, float], prev_prices: dict[str, float]) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold white", min_width=10)
    t.add_column(justify="right", min_width=14)
    t.add_column(min_width=6)

    for symbol, price in prices.items():
        prev  = prev_prices.get(symbol, price)
        change_pct = ((price - prev) / prev * 100) if prev else 0.0
        color = "green" if change_pct >= 0 else "red"
        arrow = "▲" if change_pct >= 0 else "▼"
        sign  = "+" if change_pct >= 0 else ""

        name = symbol.replace("USDT", "/USDT")
        t.add_row(
            f"[bold white]{name}[/]",
            f"[bold cyan]${price:>12,.2f}[/]",
            f"[{color}]{arrow} {sign}{change_pct:.2f}%[/]",
        )

    return Panel(t, title="[bold cyan]Live Prices[/]", border_style="cyan", height=10)


def _signals_panel(signals: dict[str, object]) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold white", min_width=12)
    t.add_column(min_width=8)
    t.add_column(min_width=16)

    direction_style = {"LONG": "bold green", "SHORT": "bold red", "NO_TRADE": "dim white"}
    direction_icon  = {"LONG": "🟢", "SHORT": "🔴", "NO_TRADE": "⚪"}

    for symbol, signal in signals.items():
        if signal is None:
            t.add_row(symbol.replace("USDT", "/USDT"), "⚪  --.--%", "[dim]Scanning...[/]")
            continue
        prob  = signal.probability
        d     = signal.direction
        color = direction_style.get(d, "white")
        icon  = direction_icon.get(d, "⚪")
        prob_color = "green" if prob >= 70 else ("red" if prob <= 30 else "yellow")
        t.add_row(
            f"[bold white]{symbol.replace('USDT', '/USDT')}[/]",
            f"[{prob_color}]{prob:.1f}%[/]",
            f"[{color}]{icon}  {d}[/]",
        )

    return Panel(t, title="[bold cyan]Signals[/]", border_style="cyan", height=10)


def _positions_panel(portfolio: Portfolio, prices: dict[str, float]) -> Panel:
    t = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
        row_styles=["", "dim"],
    )
    t.add_column("ID",       style="dim",          width=9)
    t.add_column("Symbol",   style="bold white",   width=10)
    t.add_column("Dir",      width=6)
    t.add_column("Entry",    justify="right",      width=12)
    t.add_column("Current",  justify="right",      width=12)
    t.add_column("Stop SL",  justify="right",      width=12)
    t.add_column("TP1",      justify="right",      width=12)
    t.add_column("TP2",      justify="right",      width=12)
    t.add_column("Unr. P&L", justify="right",      width=14)
    t.add_column("Prob",     justify="right",      width=7)
    t.add_column("Status",   width=12)

    if not portfolio.positions:
        t.add_row("─" * 5, "[dim]No open positions[/]", *["─"] * 9)
    else:
        for pos in portfolio.positions.values():
            price   = prices.get(pos.symbol, pos.entry_price)
            pnl     = pos.unrealized_pnl(price)
            pnl_pct = pos.unrealized_pnl_pct(price)
            color   = "green" if pnl >= 0 else "red"
            sign    = "+" if pnl >= 0 else ""
            d_color = "green" if pos.direction == "LONG" else "red"
            status  = "🔄 OPEN"

            t.add_row(
                pos.id,
                pos.symbol.replace("USDT", "/USDT"),
                f"[{d_color}]{pos.direction}[/]",
                f"${pos.entry_price:>10,.2f}",
                f"[cyan]${price:>10,.2f}[/]",
                f"[red]${pos.stop_loss:>10,.2f}[/]",
                f"${pos.take_profit1:>10,.2f}",
                f"${pos.take_profit2:>10,.2f}",
                f"[{color}]{sign}${pnl:>8,.2f} ({sign}{pnl_pct:.2f}%)[/]",
                f"{pos.probability:.0f}%",
                status,
            )

    return Panel(t, title="[bold cyan]Open Positions[/]", border_style="cyan")


def _trades_panel(ledger: Ledger, n: int = 8) -> Panel:
    t = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
        row_styles=["", "dim"],
    )
    t.add_column("Time",     width=20)
    t.add_column("Symbol",   width=10)
    t.add_column("Dir",      width=6)
    t.add_column("Type",     width=14)
    t.add_column("Entry",    justify="right", width=12)
    t.add_column("Exit",     justify="right", width=12)
    t.add_column("P&L",      justify="right", width=14)
    t.add_column("Result",   width=10)

    records = ledger.recent(n)
    if not records:
        t.add_row(*["─"] * 2, "[dim]No completed trades yet[/]", *["─"] * 5)
    else:
        for r in reversed(records):
            pnl_color = "green" if r.pnl >= 0 else "red"
            sign      = "+" if r.pnl >= 0 else ""
            d_color   = "green" if r.direction == "LONG" else "red"
            t.add_row(
                r.exit_time.strftime("%Y-%m-%d %H:%M"),
                r.symbol.replace("USDT", "/USDT"),
                f"[{d_color}]{r.direction}[/]",
                r.close_type.replace("_", " "),
                f"${r.entry_price:>10,.2f}",
                f"${r.exit_price:>10,.2f}",
                f"[{pnl_color}]{sign}${r.pnl:>8,.4f}[/]",
                f"[{pnl_color}]{r.result_label}[/]",
            )

    return Panel(t, title="[bold cyan]Recent Trades[/]", border_style="cyan")


def _status_bar(next_check_in: int, last_signal_time: Optional[str] = None) -> Panel:
    text = Text(justify="center")
    text.append("⏳  Next price check in  ", style="dim")
    text.append(f"{next_check_in}s", style="bold yellow")
    if last_signal_time:
        text.append(f"    │    Last signal scan: {last_signal_time}", style="dim")
    text.append("    │    Press [bold]Ctrl+C[/bold] to stop gracefully", style="dim")
    return Panel(Align.center(text), style="dim", height=3)


# ─────────────────────────────────────────────────────────────────
#  Dashboard builder
# ─────────────────────────────────────────────────────────────────

def build_layout(
    portfolio:        Portfolio,
    prices:           dict[str, float],
    prev_prices:      dict[str, float],
    signals:          dict[str, object],
    ledger:           Ledger,
    start_time:       datetime,
    next_check_in:    int,
    last_signal_time: Optional[str] = None,
) -> Layout:
    """Build the complete dashboard layout for one render frame."""
    layout = Layout()
    layout.split_column(
        Layout(name="header",    size=3),
        Layout(name="top_row",   size=10),
        Layout(name="positions"),
        Layout(name="trades"),
        Layout(name="status",    size=3),
    )
    layout["top_row"].split_row(
        Layout(name="portfolio", ratio=1),
        Layout(name="prices",    ratio=1),
        Layout(name="signals",   ratio=1),
    )

    layout["header"].update(_header(portfolio, prices, start_time))
    layout["portfolio"].update(_portfolio_panel(portfolio, prices))
    layout["prices"].update(_prices_panel(prices, prev_prices))
    layout["signals"].update(_signals_panel(signals))
    layout["positions"].update(_positions_panel(portfolio, prices))
    layout["trades"].update(_trades_panel(ledger))
    layout["status"].update(_status_bar(next_check_in, last_signal_time))

    return layout
