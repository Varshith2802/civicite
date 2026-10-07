"""Redact Swedish personal data before text reaches an LLM, a log or a trace."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

_PNR = re.compile(r"(?<![\d-])((?:19|20)?\d{2})(\d{2})(\d{2})([-+]?)(\d{4})(?![\d-])")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_PHONE = re.compile(r"(?<![\d+])(?:\+46|0046|0)[\s-]?7[02369](?:[\s-]?\d){7}(?!\d)|(?<![\d+])\+46(?:[\s-]?\d){7,10}(?!\d)")
_IBAN = re.compile(r"\bSE\d{2}(?:\s?\d{4}){5}\b", re.I)
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def classify_identity_number(text: str) -> str | None:
    """'personnummer', 'samordningsnummer' or None for a 10/12-digit candidate."""
    m = _PNR.fullmatch(text.strip())
    if not m:
        return None
    yy, mm, dd, _sep, tail = m.groups()
    ten = yy[-2:] + mm + dd + tail
    if not luhn_ok(ten):
        return None
    month, day = int(mm), int(dd)
    if not 1 <= month <= 12:
        return None
    kind = "personnummer"
    if day > 60:
        day -= 60
        kind = "samordningsnummer"
    year = int(yy) if len(yy) == 4 else 2000  # 2000 is a leap year, so 29 Feb is accepted for YY
    try:
        date(year, month, day)
    except ValueError:
        return None
    return kind


@dataclass
class Redaction:
    text: str
    found: dict[str, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.found)


def redact(text: str) -> Redaction:
    found: dict[str, int] = {}

    def bump(kind: str) -> None:
        found[kind] = found.get(kind, 0) + 1

    def pnr(m: re.Match) -> str:
        kind = classify_identity_number(m.group(0))
        if kind is None:
            return m.group(0)
        bump(kind)
        return f"[{kind.upper()}]"

    out = _PNR.sub(pnr, text)
    out = _IBAN.sub(lambda m: (bump("iban"), "[IBAN]")[1], out)

    def card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and luhn_ok(digits):
            bump("card")
            return "[CARD]"
        return m.group(0)

    out = _CARD.sub(card, out)
    out = _EMAIL.sub(lambda m: (bump("email"), "[EMAIL]")[1], out)
    out = _PHONE.sub(lambda m: (bump("phone"), "[PHONE]")[1], out)
    return Redaction(out, found)
