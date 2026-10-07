"""Hybrid retrieval: BM25 (always) + optional multilingual dense embeddings, fused with RRF."""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from ..ingest.loader import Chunk
from ..text import GLOSSARY_SV_EN, content_terms, stem, tokenize

DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-small"


@dataclass
class Hit:
    chunk: Chunk
    score: float          # fused score (for ranking only)
    bm25: float
    coverage: float       # IDF-weighted share of query terms found in the chunk (0..1), used for abstention
    dense: float | None = None


class BM25:
    def __init__(self, docs_terms: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(t) for t in docs_terms]
        self.len = [len(t) for t in docs_terms]
        self.avgdl = (sum(self.len) / len(self.len)) if self.len else 0.0
        df: Counter = Counter()
        for t in docs_terms:
            df.update(set(t))
        n = len(docs_terms)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, q_terms: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for w in q_terms:
                f = tf.get(w)
                if not f:
                    continue
                s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


class Index:
    def __init__(self, chunks: list[Chunk], dense_model: str | None = None, embeddings=None):
        self.chunks = chunks
        self.terms = [content_terms(c.header() + "\n" + c.text) for c in chunks]
        self.bm25 = BM25(self.terms)
        self.term_sets = [set(t) for t in self.terms]
        self.dense_model_name = dense_model
        self.embeddings = embeddings  # numpy array (n, d), L2-normalised, or None
        self._encoder = None

    # ------------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "chunks.json").write_text(
            json.dumps({"dense_model": self.dense_model_name, "chunks": [asdict(c) for c in self.chunks]},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        if self.embeddings is not None:
            import numpy as np
            np.save(directory / "embeddings.npy", self.embeddings)

    @classmethod
    def load(cls, directory: Path) -> "Index":
        data = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in data["chunks"]]
        emb = None
        if (directory / "embeddings.npy").exists():
            import numpy as np
            emb = np.load(directory / "embeddings.npy")
        return cls(chunks, data.get("dense_model"), emb)

    # ------------------------------------------------------------------ dense (optional)
    def _encode(self, texts: list[str], query: bool):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer  # optional dependency
            self._encoder = SentenceTransformer(self.dense_model_name)
        prefix = "query: " if query else "passage: "  # e5 convention; harmless for other models
        return self._encoder.encode([prefix + t for t in texts], normalize_embeddings=True)

    def build_dense(self, model: str = DEFAULT_DENSE_MODEL) -> None:
        self.dense_model_name = model
        self.embeddings = self._encode([c.header() + "\n" + c.text for c in self.chunks], query=False)

    # ------------------------------------------------------------------ search
    def weighted_coverage(self, qmap: dict[str, set[str]], chunk_terms: set[str]) -> float:
        """IDF-weighted share of query terms present in the chunk.

        Terms that never occur in the collection get the maximum IDF, so questions about things the
        collection does not cover ("opening hours in Malmö") score low even if generic words match.
        A term counts as present if it or one of its glossary translations occurs in the chunk.
        """
        if not qmap:
            return 0.0
        n = max(1, len(self.chunks))
        max_idf = math.log(1 + (n + 0.5) / 0.5)
        total = covered = 0.0
        for term, alts in qmap.items():
            known = [self.bm25.idf[a] for a in alts if a in self.bm25.idf]
            w = min(known) if known else max_idf
            total += w
            if alts & chunk_terms:
                covered += w
        return covered / total if total else 0.0

    def query_terms(self, query: str) -> dict[str, set[str]]:
        """Original query term -> set of acceptable index terms (itself + glossary translations)."""
        out: dict[str, set[str]] = {}
        for tok in tokenize(query):
            terms = content_terms(tok)
            if not terms:
                continue
            alts = set(terms)
            gloss = GLOSSARY_SV_EN.get(tok)
            if gloss:
                alts |= set(content_terms(gloss))
            out.setdefault(terms[0], set()).update(alts)
        return out

    def search(self, query: str, k: int = 5, rrf_k: int = 60) -> list[Hit]:
        qmap = self.query_terms(query)
        q = sorted({t for alts in qmap.values() for t in alts})
        if not self.chunks:
            return []
        bm = self.bm25.scores(q)
        rank_lists = [sorted(range(len(bm)), key=lambda i: -bm[i])]
        dense_scores = None
        if self.embeddings is not None and self.dense_model_name:
            try:
                qv = self._encode([query], query=True)[0]
                dense_scores = (self.embeddings @ qv).tolist()
                rank_lists.append(sorted(range(len(dense_scores)), key=lambda i: -dense_scores[i]))
            except ImportError:
                dense_scores = None
        fused = [0.0] * len(self.chunks)
        for ranking in rank_lists:
            for r, i in enumerate(ranking):
                fused[i] += 1.0 / (rrf_k + r + 1)
        order = sorted(range(len(fused)), key=lambda i: -fused[i])
        hits = []
        for i in order[:k]:
            if bm[i] <= 0 and (dense_scores is None or dense_scores[i] < 0.75):
                continue
            cov = self.weighted_coverage(qmap, self.term_sets[i])
            hits.append(Hit(self.chunks[i], fused[i], bm[i], cov,
                            None if dense_scores is None else float(dense_scores[i])))
        return hits


def confidence(hits: list[Hit]) -> float:
    """0..1 evidence strength used for abstention (lexical coverage, or dense similarity if higher)."""
    if not hits:
        return 0.0
    best = max(h.coverage for h in hits[:3])
    dense = max((h.dense for h in hits[:3] if h.dense is not None), default=None)
    if dense is not None:
        # e5 similarities for related text are ~0.8-0.9; map 0.78..0.90 onto 0..1
        best = max(best, min(1.0, max(0.0, (dense - 0.78) / 0.12)))
    return best
