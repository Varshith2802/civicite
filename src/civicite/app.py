"""Factory helpers shared by the CLI, the API server and the tests."""
from __future__ import annotations

import os
from pathlib import Path

from .agent.core import CiviCite
from .agent.llm import OpenAICompatLLM
from .index.search import Index
from .ingest.loader import chunk_document, load_directory

DEFAULT_INDEX = Path(os.environ.get("CIVICITE_INDEX", ".civicite/index"))
DEFAULT_DOCS = Path(os.environ.get("CIVICITE_DOCS", "data/demo"))
DEFAULT_TRACES = Path(os.environ.get("CIVICITE_TRACES", ".civicite/traces.jsonl"))


def build_index(docs_dir: Path = DEFAULT_DOCS, out_dir: Path | None = DEFAULT_INDEX, dense: bool = False,
                dense_model: str | None = None, max_words: int = 160) -> Index:
    docs = load_directory(docs_dir)
    if not docs:
        raise SystemExit(f"no documents found in {docs_dir}")
    chunks = [c for d in docs for c in chunk_document(d, max_words=max_words)]
    index = Index(chunks)
    if dense:
        from .index.search import DEFAULT_DENSE_MODEL
        index.build_dense(dense_model or DEFAULT_DENSE_MODEL)
    if out_dir is not None:
        index.save(out_dir)
    return index


def load_or_build_index(index_dir: Path = DEFAULT_INDEX, docs_dir: Path = DEFAULT_DOCS) -> Index:
    if (index_dir / "chunks.json").exists():
        return Index.load(index_dir)
    return build_index(docs_dir, index_dir)


def make_llm_from_env() -> OpenAICompatLLM | None:
    if not (os.environ.get("CIVICITE_LLM_BASE_URL") or os.environ.get("CIVICITE_LLM_MODEL")):
        return None
    return OpenAICompatLLM()


def make_app(index: Index | None = None, llm=None, trace: bool = True, **kwargs) -> CiviCite:
    return CiviCite(index or load_or_build_index(), llm=llm, trace_path=DEFAULT_TRACES if trace else None, **kwargs)
