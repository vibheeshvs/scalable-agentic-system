"""Hybrid (BM25 + optional dense) retrieval, used for both tools and knowledge-base chunks."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..registry.models import Catalog, ToolSpec
from .dense import Embedder
from .text import tokenize

try:  # tracing is optional; traceable is a no-op when LANGSMITH_TRACING is off
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*a, **k):
        return (lambda f: f) if not a or not callable(a[0]) else a[0]


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        toks = [tokenize(d) for d in docs]
        self.n = len(toks)
        self.dl = np.array([len(t) for t in toks], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if self.n else 1.0
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        tmp: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, t in enumerate(toks):
            for term, tf in Counter(t).items():
                tmp[term].append((i, tf))
        for term, lst in tmp.items():
            ids, tfs = zip(*lst)
            self.postings[term] = (np.array(ids), np.array(tfs, dtype=np.float32))

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            ids, tf = self.postings[term]
            idf = math.log(1 + (self.n - len(ids) + 0.5) / (len(ids) + 0.5))
            denom = tf + self.k1 * (1 - self.b + self.b * self.dl[ids] / self.avgdl)
            s[ids] += idf * tf * (self.k1 + 1) / denom
        return s


class HybridIndex:
    """Lexical + dense, fused with Reciprocal Rank Fusion (no score calibration needed)."""

    def __init__(self, lexical_docs: Sequence[str], dense_docs: Sequence[str] | None = None,
                 embedder: Embedder | None = None, rrf_k: int = 60):
        self.bm25 = BM25(lexical_docs)
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.vectors = embedder.embed(list(dense_docs or lexical_docs)) if embedder else None

    def rank(self, query: str, mask: np.ndarray | None = None, mode: str = "hybrid", depth: int = 100):
        """Return [(doc_idx, fused_score, lexical_score, dense_score)] best first."""
        lex = self.bm25.scores(query)
        dense = None
        if self.vectors is not None and mode in ("hybrid", "dense"):
            q = self.embedder.embed([query])[0]
            dense = self.vectors @ q
        if mask is not None:
            lex = np.where(mask, lex, -1.0)
            if dense is not None:
                dense = np.where(mask, dense, -2.0)

        if mode == "lexical" or dense is None:
            order = np.argsort(-lex)[:depth]
            return [(int(i), float(lex[i]), float(lex[i]), None) for i in order if lex[i] > 0]
        if mode == "dense":
            order = np.argsort(-dense)[:depth]
            return [(int(i), float(dense[i]), float(lex[i]), float(dense[i])) for i in order if dense[i] > -2]

        fused: dict[int, float] = defaultdict(float)
        for r, i in enumerate(np.argsort(-lex)[:depth]):
            if lex[i] > 0:
                fused[int(i)] += 1.0 / (self.rrf_k + r + 1)
        for r, i in enumerate(np.argsort(-dense)[:depth]):
            if dense[i] > -2:
                fused[int(i)] += 1.0 / (self.rrf_k + r + 1)
        order = sorted(fused, key=lambda i: -fused[i])
        return [(i, fused[i], float(lex[i]), float(dense[i])) for i in order]


@dataclass
class ToolHit:
    tool: ToolSpec
    score: float
    lexical: float
    dense: float | None = None
    reasons: list[str] = field(default_factory=list)


def _trace_hits(hits: list[ToolHit]) -> dict:
    """What LangSmith shows for a retrieval: the tool cards and their scores, not whole schemas."""
    return {"documents": [{"page_content": h.tool.card(), "type": "Document",
                           "metadata": {"id": h.tool.id, "score": round(h.score, 4), "lexical": round(h.lexical, 3),
                                        "dense": None if h.dense is None else round(h.dense, 3)}} for h in hits]}


class ToolIndex:
    """Search the catalog for the few tools a step actually needs."""

    def __init__(self, catalog: Catalog, embedder: Embedder | None = None):
        self.catalog = catalog
        self.tools: list[ToolSpec] = [t for t in catalog.tools.values() if t.kind == "http"]
        self.services = np.array([t.service for t in self.tools])
        self.index = HybridIndex([self._lexical_doc(t) for t in self.tools],
                                 [t.embed_text() for t in self.tools], embedder)

    def _lexical_doc(self, t: ToolSpec) -> str:
        svc = self.catalog.services.get(t.service)
        return f"{t.service} {svc.title if svc else ''} {t.search_text()}"

    def detect_services(self, text: str) -> list[str]:
        """Explicit service mentions ("...in Stripe", "my PayPal balance")."""
        words = set(re.findall(r"[a-z0-9_-]+", text.lower()))  # "...in PayPal?" must still match
        return [s for s in self.catalog.services if s.lower() in words]

    @traceable(run_type="retriever", name="tool_retrieval", process_inputs=lambda i: {
        "query": i.get("query"), "k": i.get("k"), "services": i.get("services")}, process_outputs=_trace_hits)
    def search(self, query: str, k: int = 8, services: Sequence[str] | None = None,
               mode: str = "hybrid") -> list[ToolHit]:
        mask = np.isin(self.services, list(services)) if services is not None else None
        ranked = self.index.rank(query, mask=mask, mode=mode)[:k]
        return [ToolHit(self.tools[i], score, lex, dense) for i, score, lex, dense in ranked]
