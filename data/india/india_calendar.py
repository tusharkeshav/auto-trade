# ─────────────────────────────────────────────────────────────────
#  data/india/india_calendar.py
#  NSE trading calendar: IST session hours + holiday detection.
#
#  IST = UTC+5:30
#  NSE cash market: 09:15–15:30 IST
#  NSE pre-open: 09:00–09:15 IST
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

from datetime import date, time, datetime, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# NSE trading session in IST
NSE_OPEN  = time(9, 15)
NSE_CLOSE = time(15, 30)

# NSE declared holidays 2025 (update yearly from NSE circular)
# Source: https://www.nseindia.com/resources/exchange-communication-holidays
NSE_HOLIDAYS_2025: frozenset[date] = frozenset([
    date(2025, 2, 26),   # Mahashivratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Id-Ul-Fitr (Ramadan Eid)
    date(2025, 4, 10),   # Shri Mahavir Jayanti
    date(2025, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 27),   # Ganesh Chaturthi
    date(2025, 10, 2),   # Gandhi Jayanti / Mahatma Gandhi
    date(2025, 10, 2),   # Gandhi Jayanti
    date(2025, 10, 21),  # Diwali Laxmi Puja
    date(2025, 10, 22),  # Diwali-Balipratipada
    date(2025, 11, 5),   # Prakash Gurpurb Sri Guru Nanak Dev Ji
    date(2025, 12, 25),  # Christmas
])

# NSE declared holidays 2026 (preliminary — verify against NSE circular)
NSE_HOLIDAYS_2026: frozenset[date] = frozenset([
    date(2026, 1, 26),   # Republic Day
    date(2026, 2, 18),   # Mahashivratri
    date(2026, 3, 20),   # Holi
    date(2026, 3, 19),   # Holi
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 9),    # Shri Ram Navami
    date(2026, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 8, 15),   # Independence Day
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 11, 11),  # Diwali Laxmi Puja
    date(2026, 12, 25),  # Christmas
])

_ALL_HOLIDAYS = NSE_HOLIDAYS_2025 | NSE_HOLIDAYS_2026


class IndiaCalendar:
    """NSE trading calendar with IST session awareness."""

    @staticmethod
    def is_trading_day(d: date | None = None) -> bool:
        """True if NSE is open on this date (Mon-Fri, not a holiday)."""
        if d is None:
            d = datetime.now(IST).date()
        if d.weekday() >= 5:   # Saturday=5, Sunday=6
            return False
        return d not in _ALL_HOLIDAYS

    @staticmethod
    def is_market_open(dt: datetime | None = None) -> bool:
        """True if NSE cash market is currently open."""
        if dt is None:
            dt = datetime.now(IST)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST)
        else:
            dt = dt.astimezone(IST)

        if not IndiaCalendar.is_trading_day(dt.date()):
            return False
        t = dt.time()
        return NSE_OPEN <= t <= NSE_CLOSE

    @staticmethod
    def seconds_to_open(dt: datetime | None = None) -> int:
        """Seconds until next NSE open. 0 if market is currently open."""
        if dt is None:
            dt = datetime.now(IST)
        if IndiaCalendar.is_market_open(dt):
            return 0

        # Find next trading day
        candidate = dt.date()
        if dt.time() >= NSE_CLOSE:
            candidate += timedelta(days=1)

        while not IndiaCalendar.is_trading_day(candidate):
            candidate += timedelta(days=1)

        open_dt = datetime(
            candidate.year, candidate.month, candidate.day,
            NSE_OPEN.hour, NSE_OPEN.minute, tzinfo=IST,
        )
        delta = open_dt - dt.astimezone(IST)
        return max(0, int(delta.total_seconds()))

    @staticmethod
    def next_trading_day(after: date | None = None) -> date:
        """Return next NSE trading day after `after` (exclusive)."""
        if after is None:
            after = datetime.now(IST).date()
        candidate = after + timedelta(days=1)
        while not IndiaCalendar.is_trading_day(candidate):
            candidate += timedelta(days=1)
        return candidate
