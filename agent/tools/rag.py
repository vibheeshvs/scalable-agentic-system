"""RAG pipeline tool: retrieve from the knowledge base, then generate a grounded, cited answer.

Chunking is by markdown heading (sections of these docs are self-contained, and a heading
makes a good citation). Retrieval reuses the same hybrid BM25 + dense index as tool search.
If nothing relevant is retrieved the tool says so instead of letting the model improvise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..retrieval.dense import Embedder
from ..retrieval.index import HybridIndex


@dataclass
class Chunk:
    source: str
    heading: str
    text: str

    @property
    def cite(self) -> str:
        return f"{self.source} > {self.heading}"


def load_chunks(folder: Path, max_chars: int = 1800) -> list[Chunk]:
    chunks: list[Chunk] = []
    for f in sorted(folder.glob("*.md")):
        title, current, buf = f.stem, f.stem, []
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = current = line[2:].strip()
                continue
            if line.startswith("## "):
                if "".join(buf).strip():
                    chunks.append(Chunk(f.name, current, "\n".join(buf).strip()))
                current, buf = f"{title} / {line[3:].strip()}", []
                continue
            buf.append(line)
        if "".join(buf).strip():
            chunks.append(Chunk(f.name, current, "\n".join(buf).strip()))
    out = []
    for c in chunks:  # split very long sections on paragraph boundaries
        if len(c.text) <= max_chars:
            out.append(c)
            continue
        part = ""
        for para in re.split(r"\n\s*\n", c.text):
            if len(part) + len(para) > max_chars and part:
                out.append(Chunk(c.source, c.heading, part.strip()))
                part = ""
            part += para + "\n\n"
        if part.strip():
            out.append(Chunk(c.source, c.heading, part.strip()))
    return out


class KnowledgeBase:
    def __init__(self, folder: Path, embedder: Embedder | None = None):
        self.chunks = load_chunks(folder)
        self.index = HybridIndex([f"{c.heading} {c.heading} {c.text}" for c in self.chunks],
                                 [f"{c.heading}. {c.text}" for c in self.chunks], embedder) if self.chunks else None

    def retrieve(self, question: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if not self.index:
            return []
        ranked = self.index.rank(question)
        # require at least some lexical overlap; pure-dense matches on tiny KBs are mostly noise
        return [(self.chunks[i], s) for i, s, lex, _ in ranked if lex > 0][:k]


RAG_SYSTEM = """You answer questions using ONLY the knowledge-base excerpts provided.
Cite the excerpt you used in square brackets, e.g. [refunds-policy.md > Refund policy / Approvals].
If the excerpts don't contain the answer, say you couldn't find it in the knowledge base. Do not guess."""


def rag_answer(kb: KnowledgeBase, llm, question: str, k: int = 4) -> dict[str, Any]:
    hits = kb.retrieve(question, k)
    if not hits:
        return {"answer": "I couldn't find anything about that in the knowledge base.", "sources": []}
    context = "\n\n".join(f"[{c.cite}]\n{c.text}" for c, _ in hits)
    answer = llm.text([SystemMessage(RAG_SYSTEM), HumanMessage(f"Question: {question}\n\nExcerpts:\n{context}")], name="rag_generate")
    return {"answer": answer, "sources": [c.cite for c, _ in hits]}
