"""The CiviCite agent: retrieve -> (LLM tool loop | extractive planner) -> verify -> trace."""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from ..guard.injection import is_suspicious
from ..guard.pii import redact
from ..index.search import Hit, Index, confidence
from ..ingest.loader import Chunk
from ..text import content_terms, sentences
from ..verify.citations import CITE, normalize_citation_placement, verify
from . import tools as T
from .llm import LLMError

SYSTEM_PROMPT = """You are CiviCite, an assistant that answers questions about Swedish public services.
Rules:
1. Use ONLY the numbered sources you are given or find with search_documents. Never use outside knowledge.
2. Put the source number in square brackets after EVERY factual sentence, e.g. "You must report within one week [2]."
3. Never do date arithmetic yourself: call calculate_deadline or next_business_day and mark sentences that state
   a computed date with [calc] in addition to the source of the rule.
4. If the sources do not answer the question, reply exactly: "I could not find this in the documents I have."
   and suggest which agency to contact.
5. Be brief (at most 5 sentences). Answer in the language of the question."""

ABSTAIN_EN = "I could not find this in the documents I have."
ABSTAIN_SV = "Jag kunde inte hitta detta i de dokument jag har."
_SV_HINT = re.compile(r"\b(hur|vad|när|vilken|vilka|måste|jag|kan|får|ska|min|mitt|mina)\b", re.I)
_DEADLINE_Q = re.compile(r"\b(deadline|last day|latest|by when|when (?:do|must|should|is)|how long|senast|sista dag|"
                         r"när måste|när ska|frist)\b", re.I)
_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
          "ten": 10, "eleven": 11, "twelve": 12, "fourteen": 14, "thirty": 30, "ninety": 90,
          "en": 1, "ett": 1, "två": 2, "tre": 3, "fyra": 4, "fem": 5, "sex": 6, "sju": 7, "åtta": 8, "nio": 9, "tio": 10}
_DURATION = re.compile(
    r"\b(?:within|no later than|inom|senast)\s+(\d+|" + "|".join(_WORDS) + r")\s+"
    r"(days?|weeks?|months?|dagar|dag|veckor|vecka|månader|månad)\b", re.I)
_FIXED = re.compile(r"\b(?:no later than|by|senast(?: den)?)\s+(\d{1,2})\s+(" + "|".join(T.MONTHS) + r")\b", re.I)
_ROLL = re.compile(r"next working day|nästa vardag|närmast följande vardag", re.I)
_YEAR = re.compile(r"\b(20\d{2})\b")


_CUES = [
    (re.compile(r"\b(how (do|can|should) i|how to|hur (gör|ansöker|anmäler) jag)\b", re.I), re.compile(r"^how\b|^hur\b", re.I)),
    (re.compile(r"\b(who|vem|vilka)\b", re.I), re.compile(r"^who\b|^vem\b", re.I)),
    (re.compile(r"\b(deadline|last day|latest|how long|by when|senast|sista dag|hur lång tid)\b", re.I),
     re.compile(r"deadline|when|time|frist|senast", re.I)),
    (re.compile(r"\b(how many|how much|hur många|hur mycket)\b", re.I), re.compile(r"how many|how much|hur många", re.I)),
]


def _cue_match(question: str, section: str) -> bool:
    """Does the section heading answer the *type* of question (how/who/deadline/how many)?"""
    return any(q_rx.search(question) and s_rx.search(section) for q_rx, s_rx in _CUES)


def _cite(sentence: str, n: int) -> str:
    """Append a citation before the sentence's final punctuation."""
    s = sentence.rstrip()
    if s and s[-1] in ".!?":
        return f"{s[:-1]} [{n}]{s[-1]}"
    return f"{s} [{n}]."


@dataclass
class Source:
    n: int
    chunk_id: str
    title: str
    agency: str
    url: str
    section: str
    text: str


@dataclass
class AskResult:
    trace_id: str
    question: str                  # after PII redaction
    answer: str
    abstained: bool
    mode: str
    confidence: float
    sources: list[Source]
    verification: dict[str, Any]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    pii_redacted: dict[str, int] = field(default_factory=dict)
    blocked_chunks: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CiviCite:
    def __init__(self, index: Index, llm=None, k: int = 5, abstain_threshold: float = 0.34,
                 min_faithfulness: float = 0.6, strict: bool = False, trace_path: Path | None = None,
                 max_steps: int = 5):
        self.index = index
        self.llm = llm
        self.k = k
        self.abstain_threshold = abstain_threshold
        self.min_faithfulness = min_faithfulness
        self.strict = strict
        self.trace_path = trace_path
        self.max_steps = max_steps

    # ------------------------------------------------------------------ public
    def ask(self, question: str, mode: str | None = None) -> AskResult:
        t0 = time.perf_counter()
        mode = mode or ("llm" if self.llm is not None else "extractive")
        red = redact(question)
        q = red.text
        hits, blocked = self._search(q, self.k)
        conf = confidence(hits)
        registry: dict[str, Source] = {}
        tool_log: list[dict[str, Any]] = []
        error = None
        if mode == "llm":
            if self.llm is None:
                raise ValueError("mode='llm' needs an LLM client")
            try:
                answer, abstained = self._llm_answer(q, hits, conf, registry, tool_log, blocked)
            except LLMError as e:
                error = f"{e} - fell back to extractive mode"
                mode = "extractive"
                registry.clear()
                tool_log.clear()
                answer, abstained = self._extractive_answer(q, hits, conf, registry, tool_log)
        else:
            answer, abstained = self._extractive_answer(q, hits, conf, registry, tool_log)

        answer = normalize_citation_placement(answer)
        cited = {int(x) for grp in CITE.findall(answer) for x in re.split(r"[,;]", grp) if x.strip()}
        by_n = {s.n: s for s in registry.values()}
        calc_results = [c["result"] for c in tool_log if c["tool"] in ("calculate_deadline", "next_business_day")
                        and isinstance(c.get("result"), dict)]
        ver = verify(answer, {n: f"{s.title}. {s.section}. {s.text}" for n, s in by_n.items()}, calc_results)
        if (not abstained and self.strict and ver.label not in ("verified", "no-claims")
                and ver.faithfulness < self.min_faithfulness):
            answer, abstained = self._abstain_text(q) + " (The draft answer could not be verified against the sources.)", True
        shown = [by_n[n] for n in sorted(cited) if n in by_n] if not abstained else []
        result = AskResult(
            trace_id=uuid.uuid4().hex[:12], question=q, answer=answer, abstained=abstained, mode=mode,
            confidence=round(conf, 3), sources=shown, verification=ver.to_dict(), tool_calls=tool_log,
            pii_redacted=red.found, blocked_chunks=blocked, latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            error=error)
        self._trace(result, hits)
        return result

    # ------------------------------------------------------------------ retrieval
    def _search(self, q: str, k: int) -> tuple[list[Hit], list[str]]:
        hits = self.index.search(q, k=k + 3)
        blocked = [h.chunk.chunk_id for h in hits if is_suspicious(h.chunk.text)]
        clean = [h for h in hits if h.chunk.chunk_id not in blocked]
        return clean[:k], blocked

    @staticmethod
    def _register(registry: dict[str, Source], chunk: Chunk) -> Source:
        if chunk.chunk_id not in registry:
            registry[chunk.chunk_id] = Source(len(registry) + 1, chunk.chunk_id, chunk.title, chunk.agency, chunk.url,
                                              chunk.section, chunk.text)
        return registry[chunk.chunk_id]

    @staticmethod
    def _abstain_text(q: str) -> str:
        return ABSTAIN_SV if _SV_HINT.search(q) and not re.search(r"\b(the|what|how|when)\b", q, re.I) else ABSTAIN_EN

    # ------------------------------------------------------------------ extractive mode (no LLM)
    def _extractive_answer(self, q: str, hits: list[Hit], conf: float, registry: dict[str, Source],
                           tool_log: list[dict[str, Any]]) -> tuple[str, bool]:
        if not hits or conf < self.abstain_threshold:
            return self._abstain_text(q) + " Please contact the relevant agency directly.", True
        planned = self._plan_deadline(q, hits, registry, tool_log)
        if planned:
            return planned, False
        idf = self.index.bm25.idf
        qmap = self.index.query_terms(q)
        scored: list[tuple[float, int, int, str]] = []
        top_bm25 = max(h.bm25 for h in hits[:4]) or 1.0
        for rank, h in enumerate(hits[:4]):
            weight = (h.coverage * (max(h.bm25, 0.0) / top_bm25) ** 0.5
                      * (1.6 if _cue_match(q, h.chunk.section) else 1.0))
            title_terms = set(content_terms(h.chunk.title))
            section_terms = set(content_terms(h.chunk.section))
            section_bonus = sum(max(idf.get(a, 0.0) for a in alts) for alts in qmap.values()
                                if alts & section_terms and not alts & title_terms)
            for pos, s in enumerate(sentences(h.chunk.text)):
                s_terms = set(content_terms(s))
                overlap = 0.0
                for alts in qmap.values():
                    hit = alts & s_terms
                    if hit:
                        # words that only repeat the document's topic say little about *which* sentence answers
                        weight = 0.5 if alts & title_terms else 1.0
                        overlap += weight * max(idf.get(a, 0.0) for a in hit)
                if overlap > 0:
                    scored.append(((overlap + 0.6 * section_bonus) * weight, rank, pos, s))
        if not scored:
            return self._abstain_text(q) + " Please contact the relevant agency directly.", True
        scored.sort(key=lambda x: -x[0])
        best = scored[0][0]
        chosen = [x for x in scored[:3] if x[0] >= 0.6 * best]
        chosen.sort(key=lambda x: (x[1], x[2]))
        parts = []
        for _score, rank, _pos, s in chosen:
            src = self._register(registry, hits[rank].chunk)
            parts.append(_cite(s, src.n))
        return " ".join(parts), False

    def _plan_deadline(self, q: str, hits: list[Hit], registry: dict[str, Source],
                       tool_log: list[dict[str, Any]]) -> str | None:
        """Rule-based tool use: event date in the question + a duration/fixed-date rule in the sources."""
        if not _DEADLINE_Q.search(q):
            return None
        dates = T.find_dates(q)
        for h in hits[:3]:
            for s in sentences(h.chunk.text):
                m = _DURATION.search(s)
                if m and dates:
                    amount = int(m.group(1)) if m.group(1).isdigit() else _WORDS[m.group(1).lower()]
                    unit = m.group(2).lower()
                    unit = {"dag": "days", "dagar": "days", "vecka": "weeks", "veckor": "weeks",
                            "månad": "months", "månader": "months"}.get(unit, unit if unit.endswith("s") else unit + "s")
                    args = {"start_date": dates[0].isoformat(), "amount": amount, "unit": unit}
                    res = T.calculate_deadline(**args)
                    tool_log.append({"tool": "calculate_deadline", "args": args, "result": res})
                    src = self._register(registry, h.chunk)
                    return (f"{_cite(s, src.n)} Counting from {T.format_date(dates[0])}, the deadline is "
                            f"{T.format_date(res['deadline'])} [calc].")
                f = _FIXED.search(s)
                year = _YEAR.search(q)
                if f and year and not dates:
                    d = date(int(year.group(1)), T.MONTHS[f.group(2).lower()], int(f.group(1)))
                    src = self._register(registry, h.chunk)
                    text = _cite(s, src.n)
                    roll = any(_ROLL.search(x) for x in sentences(h.chunk.text))
                    if roll:
                        res = T.next_business_day(d)
                        tool_log.append({"tool": "next_business_day", "args": {"day": d.isoformat()}, "result": res})
                        if res["moved_because"]:
                            text += (f" In {d.year}, {T.format_date(d)} is not a working day, so the deadline is "
                                     f"{T.format_date(res['date'])} [calc].")
                        else:
                            text += f" In {d.year} that is {T.format_date(d)} [calc]."
                    return text
        return None

    # ------------------------------------------------------------------ LLM mode
    def _llm_answer(self, q: str, hits: list[Hit], conf: float, registry: dict[str, Source],
                    tool_log: list[dict[str, Any]], blocked: list[str]) -> tuple[str, bool]:
        if not hits or conf < self.abstain_threshold * 0.6:
            return self._abstain_text(q) + " Please contact the relevant agency directly.", True
        for h in hits:
            self._register(registry, h.chunk)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {q}\n\nToday's date: {date.today().isoformat()}\n\n"
                                        f"Sources:\n{self._format_sources(list(registry.values()))}"},
        ]
        for _step in range(self.max_steps):
            msg = self.llm.chat(messages, tools=T.TOOL_SCHEMAS)
            calls = msg.get("tool_calls") or []
            if not calls:
                answer = (msg.get("content") or "").strip()
                abstained = (not answer) or bool(re.search(r"could not find this|kunde inte hitta", answer, re.I))
                return answer or self._abstain_text(q), abstained
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for call in calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                try:
                    args = fn.get("arguments") or {}
                    args = json.loads(args) if isinstance(args, str) else args
                    result = self._run_tool(name, args, registry, blocked)
                except Exception as e:  # tool errors go back to the model
                    args, result = {}, {"error": f"{type(e).__name__}: {e}"}
                tool_log.append({"tool": name, "args": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": call.get("id", name), "name": name,
                                 "content": json.dumps(result, ensure_ascii=False)})
        messages.append({"role": "user", "content": "Give your final answer now, with citations."})
        msg = self.llm.chat(messages, tools=None)
        answer = (msg.get("content") or "").strip()
        return answer or self._abstain_text(q), not answer

    def _run_tool(self, name: str, args: dict[str, Any], registry: dict[str, Source], blocked: list[str]) -> Any:
        if name == "search_documents":
            query = redact(str(args.get("query", ""))).text
            hits, b = self._search(query, max(1, min(int(args.get("k", 5)), 8)))
            blocked.extend(x for x in b if x not in blocked)
            return [{"source": self._register(registry, h.chunk).n, "title": h.chunk.header(),
                     "agency": h.chunk.agency, "text": h.chunk.text} for h in hits]
        if name == "calculate_deadline":
            return T.calculate_deadline(str(args["start_date"]), int(args["amount"]), str(args["unit"]),
                                        bool(args.get("roll_to_business_day", False)))
        if name == "next_business_day":
            return T.next_business_day(str(args["day"]))
        raise ValueError(f"unknown tool {name}")

    @staticmethod
    def _format_sources(sources: list[Source]) -> str:
        return "\n\n".join(f"[{s.n}] {s.title} > {s.section} ({s.agency})\n{s.text}" for s in sources)

    # ------------------------------------------------------------------ tracing
    def _trace(self, r: AskResult, hits: list[Hit]) -> None:
        if not self.trace_path:
            return
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "trace_id": r.trace_id, "mode": r.mode,
               "question": r.question, "pii_redacted": r.pii_redacted, "confidence": r.confidence,
               "retrieved": [{"chunk": h.chunk.chunk_id, "bm25": round(h.bm25, 3), "coverage": round(h.coverage, 3)}
                             for h in hits],
               "blocked_chunks": r.blocked_chunks, "abstained": r.abstained, "tool_calls": r.tool_calls,
               "verification": {"label": r.verification["label"], "faithfulness": r.verification["faithfulness"]},
               "latency_ms": r.latency_ms, "error": r.error}
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
