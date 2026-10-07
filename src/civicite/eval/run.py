"""Evaluation harness: retrieval, answer accuracy, abstention, faithfulness, citations, tool use, latency."""
from __future__ import annotations

import json
import statistics
from importlib import resources
from pathlib import Path
from typing import Any

from ..agent.core import CiviCite
from ..agent.tools import format_date
from ..guard.pii import redact


def load_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8") if path else \
        resources.files("civicite.eval").joinpath("dataset.jsonl").read_text(encoding="utf-8")
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate(app: CiviCite, rows: list[dict[str, Any]], mode: str, k: int = 5) -> dict[str, Any]:
    per: list[dict[str, Any]] = []
    for row in rows:
        res = app.ask(row["question"], mode=mode)
        hits, _ = app._search(redact(row["question"]).text, k)
        docs = [h.chunk.doc_id for h in hits]
        exp = set(row["expected_docs"])
        rank = next((i + 1 for i, d in enumerate(docs) if d in exp), None)
        answer_l = res.answer.lower()
        correct = (row["answerable"] and not res.abstained
                   and all(m.lower() in answer_l for m in row["must_include"]))
        cited_docs = [s.chunk_id.split("#")[0] for s in res.sources]
        date_ok = None
        if row.get("expected_date"):
            date_ok = format_date(row["expected_date"]).split(" ", 1)[1].lower() in answer_l
        per.append({
            "id": row["id"], "question": row["question"], "answerable": row["answerable"],
            "abstained": res.abstained, "correct": correct, "retrieval_rank": rank,
            "faithfulness": res.verification["faithfulness"], "verification": res.verification["label"],
            "cited_docs": cited_docs, "citation_hits": sum(d in exp for d in cited_docs),
            "date_ok": date_ok, "latency_ms": res.latency_ms, "confidence": res.confidence,
            "answer": res.answer, "error": res.error,
        })
    return summarize(per, k) | {"mode": mode, "questions": per}


def summarize(per: list[dict[str, Any]], k: int = 5) -> dict[str, Any]:
    ans = [p for p in per if p["answerable"]]
    unans = [p for p in per if not p["answerable"]]
    answered = [p for p in ans if not p["abstained"]]
    tp = sum(p["abstained"] for p in unans)            # correctly abstained
    fp = sum(p["abstained"] for p in ans)              # abstained although answerable
    fn = sum(not p["abstained"] for p in unans)        # answered although not answerable
    prec, rec, f1 = _f1(tp, fp, fn)
    cites = sum(len(p["cited_docs"]) for p in answered)
    dated = [p for p in per if p["date_ok"] is not None]
    lat = sorted(p["latency_ms"] for p in per)
    return {
        "n": len(per), "answerable": len(ans), "unanswerable": len(unans),
        f"retrieval_recall@{k}": round(sum(p["retrieval_rank"] is not None for p in ans) / max(1, len(ans)), 3),
        "retrieval_mrr": round(sum(1 / p["retrieval_rank"] for p in ans if p["retrieval_rank"]) / max(1, len(ans)), 3),
        "answer_accuracy": round(sum(p["correct"] for p in ans) / max(1, len(ans)), 3),
        "abstain_precision": round(prec, 3), "abstain_recall": round(rec, 3), "abstain_f1": round(f1, 3),
        "faithfulness": round(statistics.mean([p["faithfulness"] for p in answered]), 3) if answered else None,
        "citation_precision": round(sum(p["citation_hits"] for p in answered) / cites, 3) if cites else None,
        "tool_date_accuracy": round(sum(bool(p["date_ok"]) for p in dated) / len(dated), 3) if dated else None,
        "latency_ms_p50": lat[len(lat) // 2] if lat else None,
        "latency_ms_p95": lat[int(0.95 * (len(lat) - 1))] if lat else None,
    }


def sweep(app: CiviCite, rows: list[dict[str, Any]], mode: str, thresholds: list[float]) -> list[dict[str, Any]]:
    out = []
    original = app.abstain_threshold
    try:
        for t in thresholds:
            app.abstain_threshold = t
            s = evaluate(app, rows, mode)
            out.append({"threshold": t, "abstain_f1": s["abstain_f1"], "answer_accuracy": s["answer_accuracy"]})
    finally:
        app.abstain_threshold = original
    return out


def to_markdown(s: dict[str, Any]) -> str:
    keys = [k for k in s if k not in ("questions", "mode")]
    lines = [f"### Evaluation ({s['mode']} mode)", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {s[k]} |" for k in keys]
    lines += ["", "| id | answerable | abstained | correct | rank | verification | answer |", "|---|---|---|---|---|---|---|"]
    for p in s["questions"]:
        a = p["answer"].replace("|", "/").replace("\n", " ")
        lines.append(f"| {p['id']} | {p['answerable']} | {p['abstained']} | {p['correct']} | {p['retrieval_rank']} | "
                     f"{p['verification']} | {a[:110]}{'...' if len(a) > 110 else ''} |")
    return "\n".join(lines)
