"""Sentence-level citation verification.

Every factual sentence in an answer must cite at least one source, and the cited source must support it:
  * lexical support: share of the sentence's content terms that occur in the cited chunk(s)
  * number consistency: every number in the sentence must appear in a cited chunk or in a tool result
Sentences marked ``[calc]`` are checked against the deterministic tool results instead.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from ..text import content_terms, sentences

CITE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
CALC = re.compile(r"\[calc\]", re.I)
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_META = re.compile(
    r"(could not find|couldn't find|cannot find|can't find|not (?:mentioned|covered|stated) in|no information|"
    r"please (?:verify|check|contact)|contact (?:the|your)|this is not (?:legal|official)|"
    r"kunde inte hitta|hittar inte|kontakta|sources?:|källor?:)", re.I)

SUPPORT_THRESHOLD = 0.5
_TRAILING = re.compile(r"([.!?])((?:\s*\[(?:\d+(?:\s*[,;]\s*\d+)*|calc)\])+)", re.I)


def normalize_citation_placement(text: str) -> str:
    """Move citation markers that follow the full stop in front of it: "rule. [1]" -> "rule [1]."."""
    return _TRAILING.sub(lambda m: m.group(2).rstrip() + m.group(1), text)


def _human_date(y: int, m: int, d: int) -> str:
    import calendar
    from datetime import date as _date
    try:
        dt = _date(y, m, d)
    except ValueError:
        return ""
    return f"{calendar.day_name[dt.weekday()]} {d} {calendar.month_name[m]} {y}"


def _norm_numbers(text: str) -> set[str]:
    out = set()
    for n in _NUM.findall(text):
        n = n.replace(",", ".")
        try:
            out.add(str(int(n)) if "." not in n else str(float(n)))
        except ValueError:
            out.add(n)
    return out


@dataclass
class SentenceCheck:
    text: str
    citations: list[int]
    is_claim: bool
    supported: bool
    support: float
    reason: str = ""


@dataclass
class Verification:
    sentences: list[SentenceCheck] = field(default_factory=list)
    faithfulness: float = 1.0       # supported claims / claims
    citation_coverage: float = 1.0  # claims with a citation / claims
    invalid_citations: list[int] = field(default_factory=list)

    @property
    def unsupported(self) -> list[str]:
        return [s.text for s in self.sentences if s.is_claim and not s.supported]

    @property
    def label(self) -> str:
        if not any(s.is_claim for s in self.sentences):
            return "no-claims"
        if self.faithfulness >= 0.999:
            return "verified"
        return "partially-verified" if self.faithfulness >= 0.5 else "unverified"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = self.label
        d["unsupported"] = self.unsupported
        return d


def verify(answer: str, sources: dict[int, str], tool_results: list[dict] | None = None) -> Verification:
    """``sources`` maps citation number -> chunk text."""
    tool_text = " ".join(str(v) for r in (tool_results or []) for v in r.values())
    for iso in re.findall(r"\d{4}-\d{2}-\d{2}", tool_text):  # add "Tuesday 17 March 2026" style forms
        y, m, d = (int(x) for x in iso.split("-"))
        tool_text += " " + _human_date(y, m, d)
    tool_nums = _norm_numbers(tool_text.replace("-", " "))
    tool_terms = set(content_terms(tool_text))
    checks: list[SentenceCheck] = []
    invalid: set[int] = set()
    for sent in sentences(normalize_citation_placement(answer)):
        cites = [int(x) for grp in CITE.findall(sent) for x in re.split(r"[,;]", grp) if x.strip()]
        has_calc = bool(CALC.search(sent))
        body = CALC.sub(" ", CITE.sub(" ", sent))
        terms = set(content_terms(body))
        is_claim = len(terms) >= 3 and not _META.search(body)
        if not is_claim:
            checks.append(SentenceCheck(sent, cites, False, True, 1.0, "not a factual claim"))
            continue
        valid = [c for c in cites if c in sources]
        invalid.update(c for c in cites if c not in sources)
        if not valid and not has_calc:
            checks.append(SentenceCheck(sent, cites, True, False, 0.0,
                                        "no valid citation" if cites else "uncited claim"))
            continue
        evidence = " ".join(sources[c] for c in valid)
        ev_terms = set(content_terms(evidence))
        support = len(terms & ev_terms) / len(terms) if terms else 0.0
        nums = _norm_numbers(body)
        allowed_nums = _norm_numbers(evidence) | (tool_nums if has_calc else set())
        missing = sorted(nums - allowed_nums)
        if has_calc and not valid:
            ok = not missing and bool(tool_results)
            reason = "matches tool result" if ok else f"numbers not in tool result: {missing}"
            checks.append(SentenceCheck(sent, cites, True, ok, 1.0 if ok else 0.0, reason))
            continue
        if has_calc:
            # date words come from the tool; the rest of the sentence must still match the cited rule
            rest = terms - tool_terms
            support = len(rest & ev_terms) / len(rest) if rest else 1.0
        threshold = 0.3 if has_calc else SUPPORT_THRESHOLD
        ok = support >= threshold and not missing
        reason = "supported" if ok else (f"numbers not in source: {missing}" if missing
                                        else f"low overlap with cited source ({support:.2f})")
        checks.append(SentenceCheck(sent, cites, True, ok, round(support, 3), reason))
    claims = [c for c in checks if c.is_claim]
    v = Verification(checks, invalid_citations=sorted(invalid))
    if claims:
        v.faithfulness = round(sum(c.supported for c in claims) / len(claims), 3)
        v.citation_coverage = round(sum(bool(c.citations) or "[calc]" in c.text.lower() for c in claims)
                                    / len(claims), 3)
    return v
