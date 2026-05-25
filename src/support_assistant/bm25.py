"""BM25 sparse retrieval (M8).

Built alongside Chroma at ingest time. The full chunk list is persisted as JSON
so we can rebuild the BM25Okapi instance in memory on startup — small corpus,
so this is fast enough.

Design notes:
- Tokenizer is deliberately simple (lowercase + word-boundary regex). For
  Stripe docs which are full of identifiers like `payment_intents`, `4242`,
  this gives BM25 a fair shot at matching exact tokens that dense retrieval
  often blurs together.
- We do NOT stem. Stemming hurts API/identifier matching (`refunds` vs
  `refund` is a meaningful distinction sometimes; for vague queries the dense
  side already handles it).
- Score normalization: dividing by the max score gives a score in [0, 1],
  cosmetic only. RRF uses ranks, not raw scores, so normalization doesn't
  affect fusion.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .chunk import Chunk
from .config import settings
from .store import Retrieved

_TOKEN_RE = re.compile(r"[a-z0-9_]{2,}")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class _Doc:
    chunk_id: str
    url: str
    title: str
    text: str


class BM25Index:
    def __init__(self) -> None:
        self._docs: list[_Doc] = []
        self._bm25: BM25Okapi | None = None

    # ----- build / persistence -----

    def build(self, chunks: list[Chunk]) -> None:
        """Build from a chunk list and persist to disk."""
        self._docs = [_Doc(c.chunk_id, c.url, c.title, c.text) for c in chunks]
        if not self._docs:
            self._bm25 = None
            self._save()
            return
        corpus = [tokenize(d.text) for d in self._docs]
        self._bm25 = BM25Okapi(corpus)
        self._save()

    def _save(self) -> None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        data = [{"chunk_id": d.chunk_id, "url": d.url, "title": d.title, "text": d.text} for d in self._docs]
        settings.bm25_path.write_text(json.dumps(data), encoding="utf-8")

    def load(self) -> bool:
        if not settings.bm25_path.exists():
            return False
        raw = json.loads(settings.bm25_path.read_text(encoding="utf-8"))
        self._docs = [_Doc(**d) for d in raw]
        if not self._docs:
            self._bm25 = None
            return True
        corpus = [tokenize(d.text) for d in self._docs]
        self._bm25 = BM25Okapi(corpus)
        return True

    # ----- query -----

    def query(self, q: str, top_k: int | None = None) -> list[Retrieved]:
        k = top_k or settings.bm25_top_k
        if self._bm25 is None and not self.load():
            return []
        if self._bm25 is None:
            return []
        tokens = tokenize(q)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(self._docs, scores), key=lambda x: float(x[1]), reverse=True
        )[:k]
        max_s = float(ranked[0][1]) if ranked and float(ranked[0][1]) > 0 else 1.0
        out: list[Retrieved] = []
        for d, s in ranked:
            s = float(s)
            if s <= 0:
                continue
            out.append(
                Retrieved(
                    chunk_id=d.chunk_id,
                    url=d.url,
                    title=d.title,
                    text=d.text,
                    score=s / max_s,
                )
            )
        return out

    @property
    def size(self) -> int:
        return len(self._docs)


# Module-level singleton — loaded on first use by retrieve().
_index: BM25Index | None = None


def get_index() -> BM25Index:
    global _index
    if _index is None:
        _index = BM25Index()
        _index.load()
    return _index


def rebuild(chunks: list[Chunk]) -> None:
    """Rebuild and replace the singleton."""
    global _index
    idx = BM25Index()
    idx.build(chunks)
    _index = idx
