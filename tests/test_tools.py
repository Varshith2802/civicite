from datetime import date

from civicite.agent.tools import (calculate_deadline, easter_sunday, find_dates, format_date, next_business_day,
                                  swedish_holidays)


def test_easter_and_holidays():
    assert easter_sunday(2026) == date(2026, 4, 5)
    assert easter_sunday(2027) == date(2027, 3, 28)
    h = swedish_holidays(2026)
    assert h[date(2026, 6, 19)].startswith("Midsummer Eve")
    assert h[date(2026, 4, 3)].startswith("Good Friday")
    assert h[date(2026, 10, 31)].startswith("All Saints")


def test_next_business_day():
    r = next_business_day("2026-05-02")  # Saturday
    assert r["date"] == "2026-05-04" and len(r["moved_because"]) == 2
    assert next_business_day("2026-12-24")["date"] == "2026-12-28"  # Christmas Eve -> after Boxing Day weekend
    assert next_business_day("2026-03-10")["moved_because"] == []


def test_calculate_deadline():
    assert calculate_deadline("2026-03-10", 1, "weeks")["deadline"] == "2026-03-17"
    assert calculate_deadline("2026-01-15", 90, "days")["deadline"] == "2026-04-15"
    assert calculate_deadline("2026-01-31", 1, "months")["deadline"] == "2026-02-28"
    r = calculate_deadline("2026-06-12", 1, "weeks", roll_to_business_day=True)
    assert r["deadline"] == "2026-06-22"  # 19 June is Midsummer Eve, then weekend


def test_find_dates_formats():
    assert find_dates("moved on 2026-03-10") == [date(2026, 3, 10)]
    assert find_dates("on 10 March 2026") == [date(2026, 3, 10)]
    assert find_dates("den 10 mars 2026") == [date(2026, 3, 10)]
    assert find_dates("March 10, 2026") == [date(2026, 3, 10)]
    assert format_date("2026-03-17") == "Tuesday 17 March 2026"
