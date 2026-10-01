"""Optional dense embeddings. The system works without them (BM25 only) and gets better with them.

Backends, picked by EMBEDDINGS env var (default "auto" = first one that works, in this order):
  - "openai"     -> text-embedding-3-small via langchain-openai (needs OPENAI_API_KEY)
  - "wordllama"  -> 16 MB static embeddings bundled in the pip package (offline)
  - "none"       -> lexical only
Embeddings are cached on disk keyed by (model, text hash), so re-indexing 1k+ tools is free
after the first run and adding a new service only embeds the new tools.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol

import numpy as np

CACHE_DIR = Path(os.getenv("EMBED_CACHE_DIR", Path(__file__).resolve().parents[2] / ".cache" / "embeddings"))


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class CachedEmbedder:
    """Wraps any embedder with a tiny on-disk cache (one .npy per text hash batch)."""

    def __init__(self, inner: Embedder):
        self.inner = inner
        self.name = inner.name
        self.dir = CACHE_DIR / hashlib.md5(inner.name.encode()).hexdigest()[:10]
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.dir / "index.json"
        self._index: dict[str, int] = json.loads(self.index_file.read_text(encoding="utf-8")) if self.index_file.exists() else {}
        self._matrix = np.load(self.dir / "vectors.npy") if (self.dir / "vectors.npy").exists() else None

    def embed(self, texts: list[str]) -> np.ndarray:
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self._index]
        if missing:
            uniq = dict(missing)
            vecs = self.inner.embed(list(uniq.values()))
            start = 0 if self._matrix is None else len(self._matrix)
            self._matrix = vecs if self._matrix is None else np.vstack([self._matrix, vecs])
            for i, k in enumerate(uniq):
                self._index[k] = start + i
            np.save(self.dir / "vectors.npy", self._matrix)
            self.index_file.write_text(json.dumps(self._index), encoding="utf-8")
        assert self._matrix is not None
        return self._matrix[[self._index[k] for k in keys]]


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        from langchain_openai import OpenAIEmbeddings  # optional dependency

        self.name = f"openai:{model}"
        self._emb = OpenAIEmbeddings(model=model)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 256):
            out.extend(self._emb.embed_documents(texts[i : i + 256]))
        return _normalize(np.array(out, dtype=np.float32))


class WordLlamaEmbedder:
    """Tiny static embedding model (16 MB, ships inside the pip wheel, CPU-only, no network).

    Good enough to catch synonyms that BM25 misses ("chargeback" ~ "dispute"). For
    production I'd use a proper sentence encoder; this is what runs offline in CI.
    """

    def __init__(self, dim: int = 256):
        import shutil

        import wordllama
        from wordllama import WordLlama

        # wordllama 0.4 looks for its bundled tokenizer in the wrong folder; copy it into the cache
        tok_dir = WordLlama.get_file_path("tokenizer", None)
        fname = "l2_supercat_tokenizer_config.json"
        if not (tok_dir / fname).exists():
            tok_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(Path(wordllama.__file__).parent / "tokenizers" / fname, tok_dir / fname)
        self.name = f"wordllama:l2_supercat_{dim}"
        self._wl = WordLlama.load(dim=dim)

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.asarray(self._wl.embed(texts, norm=True), dtype=np.float32))


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1
    return m / n


def get_embedder(kind: str | None = None) -> Embedder | None:
    kind = (kind or os.getenv("EMBEDDINGS", "auto")).lower()
    if kind == "none":
        return None
    if kind in ("openai", "auto") and os.getenv("OPENAI_API_KEY"):
        try:
            return CachedEmbedder(OpenAIEmbedder())
        except Exception:
            if kind == "openai":
                raise
    for name, cls in (("wordllama", WordLlamaEmbedder),):
        if kind in (name, "auto"):
            try:
                return CachedEmbedder(cls())
            except Exception:
                if kind == name:
                    raise
    return None
