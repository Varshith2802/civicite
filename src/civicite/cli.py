"""civicite index | fetch | ask | serve | eval"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from .agent.llm import OpenAICompatLLM


def _llm_from_args(a) -> OpenAICompatLLM | None:
    if a.mode == "extractive":
        return None
    if a.llm_base_url or a.llm_model or a.mode == "llm":
        return OpenAICompatLLM(base_url=a.llm_base_url, model=a.llm_model)
    from .app import make_llm_from_env
    return make_llm_from_env()


def _add_llm(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", choices=["auto", "extractive", "llm"], default="auto",
                   help="extractive = no LLM (deterministic); llm = OpenAI-compatible model with tools")
    p.add_argument("--llm-base-url", help="e.g. http://localhost:11434/v1 (Ollama) or https://api.openai.com/v1")
    p.add_argument("--llm-model", help="e.g. llama3.1:8b, qwen2.5:7b-instruct, gpt-4o-mini")
    p.add_argument("--index", type=Path, default=None)
    p.add_argument("--strict", action="store_true", help="replace unverifiable answers with an abstention")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="civicite", description="Verified RAG assistant for Swedish public services")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="build the search index from a folder of documents")
    p.add_argument("--docs", type=Path, default=Path("data/demo"))
    p.add_argument("--out", type=Path, default=Path(".civicite/index"))
    p.add_argument("--dense", action="store_true", help="add multilingual embeddings (pip install .[dense])")
    p.add_argument("--dense-model", default=None)

    p = sub.add_parser("fetch", help="crawl official pages into Markdown documents")
    p.add_argument("--seed", action="append", required=True)
    p.add_argument("--prefix", action="append", help="only follow URLs starting with this (default: the seeds)")
    p.add_argument("--agency", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-pages", type=int, default=100)
    p.add_argument("--delay", type=float, default=1.0)

    p = sub.add_parser("ask", help="ask a question")
    p.add_argument("question")
    p.add_argument("--json", action="store_true")
    _add_llm(p)

    p = sub.add_parser("serve", help="run the web UI + JSON API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    _add_llm(p)

    p = sub.add_parser("eval", help="run the evaluation suite")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("eval_results"))
    p.add_argument("--sweep", action="store_true", help="sweep abstention thresholds")
    p.add_argument("--min-recall", type=float, default=0.0)
    p.add_argument("--min-accuracy", type=float, default=0.0)
    p.add_argument("--min-faithfulness", type=float, default=0.0)
    p.add_argument("--min-abstain-f1", type=float, default=0.0)
    _add_llm(p)

    a = ap.parse_args(argv)

    if a.cmd == "index":
        from .app import build_index
        idx = build_index(a.docs, a.out, dense=a.dense, dense_model=a.dense_model)
        print(f"indexed {len(idx.chunks)} chunks from {a.docs} -> {a.out}"
              + (f" (+ dense: {idx.dense_model_name})" if idx.embeddings is not None else ""))
        return 0

    if a.cmd == "fetch":
        from .ingest.fetch import crawl
        files = crawl(a.seed, a.prefix or a.seed, a.out, a.agency, a.max_pages, a.delay)
        print(f"saved {len(files)} pages to {a.out}. Next: civicite index --docs {a.out}")
        return 0

    from .app import DEFAULT_INDEX, load_or_build_index, make_app
    index = load_or_build_index(a.index or DEFAULT_INDEX)
    llm = _llm_from_args(a)
    app = make_app(index, llm=llm, strict=a.strict)
    mode = None if a.mode == "auto" else a.mode

    if a.cmd == "ask":
        r = app.ask(a.question, mode=mode)
        if a.json:
            print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
            return 0
        print("\n" + textwrap.fill(r.answer, 100) + "\n")
        for s in r.sources:
            print(f"  [{s.n}] {s.title} > {s.section} - {s.agency} {s.url}")
        v = r.verification
        print(f"\nverification: {v['label']} (faithfulness {v['faithfulness']}, citation coverage "
              f"{v['citation_coverage']}) | mode: {r.mode} | confidence: {r.confidence} | {r.latency_ms} ms")
        if r.pii_redacted:
            print(f"personal data removed before processing: {r.pii_redacted}")
        if v["unsupported"]:
            print("unsupported sentences:\n  - " + "\n  - ".join(v["unsupported"]))
        if r.error:
            print(f"note: {r.error}", file=sys.stderr)
        return 0

    if a.cmd == "serve":
        import uvicorn
        from .api.server import create_app
        print(f"CiviCite on http://{a.host}:{a.port}  (mode: {a.mode}, llm: {getattr(llm, 'name', None)})")
        uvicorn.run(create_app(app, default_mode=mode), host=a.host, port=a.port, log_level="info")
        return 0

    if a.cmd == "eval":
        from .eval.run import evaluate, load_dataset, sweep, to_markdown
        rows = load_dataset(a.dataset)
        eff_mode = mode or ("llm" if llm else "extractive")
        app.trace_path = None
        s = evaluate(app, rows, eff_mode)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"eval_{eff_mode}.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
        md = to_markdown(s)
        if a.sweep:
            sw = sweep(app, rows, eff_mode, [0.1, 0.2, 0.25, 0.3, 0.34, 0.4, 0.5, 0.6])
            md += "\n\n### Abstention threshold sweep\n\n| threshold | abstain_f1 | answer_accuracy |\n|---|---|---|\n"
            md += "\n".join(f"| {r['threshold']} | {r['abstain_f1']} | {r['answer_accuracy']} |" for r in sw)
        (a.out / f"eval_{eff_mode}.md").write_text(md + "\n", encoding="utf-8")
        print(md)
        gates = [("retrieval_recall@5", a.min_recall), ("answer_accuracy", a.min_accuracy),
                 ("faithfulness", a.min_faithfulness), ("abstain_f1", a.min_abstain_f1)]
        failed = [(k, s[k], v) for k, v in gates if v and (s[k] is None or s[k] < v)]
        for k, got, want in failed:
            print(f"GATE FAILED: {k} = {got} < {want}", file=sys.stderr)
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
