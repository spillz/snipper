from __future__ import annotations

import calendar
from datetime import datetime, timezone


def parse_date(s: str, fmt: str) -> float:
    dt = datetime.strptime(s.strip(), fmt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def parse_date_safe(s: str, fmt: str, *, default: str = "2000") -> float:
    s = (s or "").strip()
    if not s:
        s = default
    return parse_date(s, fmt)


def format_date(ts: float, fmt: str) -> str:
    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return dt.strftime(fmt)


def last_day_of_month(year: int, month: int) -> int:
    return int(calendar.monthrange(int(year), int(month))[1])


def add_months(dt: datetime, months: int) -> datetime:
    if months == 0:
        return dt
    total = (dt.year * 12) + (dt.month - 1) + int(months)
    year = total // 12
    month = (total % 12) + 1
    day = min(dt.day, last_day_of_month(year, month))
    return dt.replace(year=year, month=month, day=day)
