"""Deterministic tools the agent can call. Date maths is never left to the language model."""
from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
MONTHS.update({"januari": 1, "februari": 2, "mars": 3, "april": 4, "maj": 5, "juni": 6, "juli": 7,
               "augusti": 8, "september": 9, "oktober": 10, "november": 11, "december": 12})


def easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def swedish_holidays(year: int) -> dict[date, str]:
    """Public holidays plus the eves that are treated as non-working days (Midsummer, Christmas, New Year's Eve)."""
    easter = easter_sunday(year)
    midsummer_eve = next(date(year, 6, d) for d in range(19, 26) if date(year, 6, d).weekday() == 4)
    all_saints = next(date(year, 10, 31) + timedelta(days=i) for i in range(7)
                      if (date(year, 10, 31) + timedelta(days=i)).weekday() == 5)
    return {
        date(year, 1, 1): "New Year's Day (nyårsdagen)",
        date(year, 1, 6): "Epiphany (trettondedag jul)",
        easter - timedelta(days=2): "Good Friday (långfredagen)",
        easter: "Easter Sunday (påskdagen)",
        easter + timedelta(days=1): "Easter Monday (annandag påsk)",
        date(year, 5, 1): "May Day (första maj)",
        easter + timedelta(days=39): "Ascension Day (Kristi himmelsfärdsdag)",
        easter + timedelta(days=49): "Whit Sunday (pingstdagen)",
        date(year, 6, 6): "National Day (Sveriges nationaldag)",
        midsummer_eve: "Midsummer Eve (midsommarafton)",
        midsummer_eve + timedelta(days=1): "Midsummer Day (midsommardagen)",
        all_saints: "All Saints' Day (alla helgons dag)",
        date(year, 12, 24): "Christmas Eve (julafton)",
        date(year, 12, 25): "Christmas Day (juldagen)",
        date(year, 12, 26): "Boxing Day (annandag jul)",
        date(year, 12, 31): "New Year's Eve (nyårsafton)",
    }


def is_working_day(d: date) -> bool:
    return d.weekday() < 5 and d not in swedish_holidays(d.year)


def next_business_day(day: str | date) -> dict:
    d = parse_date(day) if isinstance(day, str) else day
    reasons = []
    while not is_working_day(d):
        reasons.append(f"{d.isoformat()} is a {swedish_holidays(d.year).get(d, WEEKDAYS[d.weekday()])}")
        d += timedelta(days=1)
    return {"date": d.isoformat(), "weekday": WEEKDAYS[d.weekday()], "moved_because": reasons}


def add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    y, m = d.year + m // 12, m % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def calculate_deadline(start_date: str, amount: int, unit: str, roll_to_business_day: bool = False) -> dict:
    """start_date + amount unit (days/weeks/months), optionally moved to the next Swedish working day."""
    d0 = parse_date(start_date)
    u = unit.lower().rstrip("s")
    if u in ("day", "dag", "dagar"):
        d = d0 + timedelta(days=amount)
    elif u in ("week", "vecka", "veckor"):
        d = d0 + timedelta(weeks=amount)
    elif u in ("month", "månad", "månader", "manad"):
        d = add_months(d0, amount)
    else:
        raise ValueError(f"unknown unit: {unit}")
    result = {"start": d0.isoformat(), "rule": f"{amount} {unit}", "deadline": d.isoformat(),
              "weekday": WEEKDAYS[d.weekday()], "moved_because": []}
    if roll_to_business_day:
        nb = next_business_day(d)
        result.update(deadline=nb["date"], weekday=nb["weekday"], moved_because=nb["moved_because"])
    return result


_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DMY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th|e|:e)?\s+(?:of\s+)?([A-Za-zåäö]+)\.?,?\s+(\d{4})\b")
_MDY = re.compile(r"\b([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")


def parse_date(text: str) -> date:
    found = find_dates(text)
    if not found:
        raise ValueError(f"no date found in {text!r}")
    return found[0]


def find_dates(text: str) -> list[date]:
    out: list[tuple[int, date]] = []
    for m in _ISO.finditer(text):
        try:
            out.append((m.start(), date(int(m.group(1)), int(m.group(2)), int(m.group(3)))))
        except ValueError:
            pass
    for m in _DMY.finditer(text):
        mon = MONTHS.get(m.group(2).lower())
        if mon:
            try:
                out.append((m.start(), date(int(m.group(3)), mon, int(m.group(1)))))
            except ValueError:
                pass
    for m in _MDY.finditer(text):
        mon = MONTHS.get(m.group(1).lower())
        if mon:
            try:
                out.append((m.start(), date(int(m.group(3)), mon, int(m.group(2)))))
            except ValueError:
                pass
    out.sort()
    seen, dates = set(), []
    for _, d in out:
        if d not in seen:
            seen.add(d)
            dates.append(d)
    return dates


def format_date(d: str | date) -> str:
    d = date.fromisoformat(d) if isinstance(d, str) else d
    return f"{WEEKDAYS[d.weekday()]} {d.day} {calendar.month_name[d.month]} {d.year}"


# JSON schemas advertised to tool-calling LLMs (OpenAI "tools" format)
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "Search the official-guidance document collection. Returns numbered sources to cite as [n].",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "search query, any language"},
            "k": {"type": "integer", "description": "number of results (1-8)", "default": 5}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculate_deadline",
        "description": "Add a period to a date (Swedish calendar). Use for any date arithmetic.",
        "parameters": {"type": "object", "properties": {
            "start_date": {"type": "string", "description": "YYYY-MM-DD"},
            "amount": {"type": "integer"},
            "unit": {"type": "string", "enum": ["days", "weeks", "months"]},
            "roll_to_business_day": {"type": "boolean", "default": False}},
            "required": ["start_date", "amount", "unit"]}}},
    {"type": "function", "function": {
        "name": "next_business_day",
        "description": "Return the date itself if it is a Swedish working day, otherwise the next working day.",
        "parameters": {"type": "object", "properties": {"day": {"type": "string", "description": "YYYY-MM-DD"}},
                       "required": ["day"]}}},
]
