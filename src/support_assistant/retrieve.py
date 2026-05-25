"""Retrieval stage. Owns: embed question → dense search → (optional) BM25 fuse → (optional) rerank.

Modes (selectable per call or via settings):
- dense-only        → vector search only
- hybrid            → dense + BM25, fused with Reciprocal Rank Fusion
- + rerank kind     → "llm" | "crossencoder" | "none"

The retrieve function returns a `RetrievalResult` exposing both the final
candidate list AND the *top dense cosine score* — needed by abstention,
because RRF fusion produces scores on a different scale than cosine and
would otherwise nuke the score-floor gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import bm25 as bm25_mod
from .config import settings
from .embed import OpenAIEmbedder
from .observability import StageTimings, stage_timer
from .rerank import get_reranker
from .store import Retrieved, VectorStore


@dataclass(frozen=True)
class RetrievalResult:
    candidates: list[Retrieved]
    top_dense_score: float   # always cosine similarity of top-1 dense, even in hybrid mode


def rrf_fuse(
    dense: list[Retrieved],
    sparse: list[Retrieved],
    k: int | None = None,
) -> list[Retrieved]:
    """Reciprocal Rank Fusion. score(d) = sum_over_rankings(1 / (k + rank)).

    k is the standard RRF damping constant (60 in the original paper). Higher k
    means rank position matters less; lower k means top results dominate.
    """
    kk = k if k is not None else settings.rrf_k
    scores: dict[str, float] = {}
    by_id: dict[str, Retrieved] = {}
    for rank, r in enumerate(dense):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (kk + rank + 1)
        by_id.setdefault(r.chunk_id, r)
    for rank, r in enumerate(sparse):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (kk + rank + 1)
        by_id.setdefault(r.chunk_id, r)
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [
        Retrieved(
            chunk_id=cid,
            url=by_id[cid].url,
            title=by_id[cid].title,
            text=by_id[cid].text,
            score=scores[cid],
        )
        for cid in ordered
    ]


def retrieve(
    question: str,
    timings: StageTimings,
    *,
    q_vec: list[float] | None = None,
    embedder: OpenAIEmbedder | None = None,
    store: VectorStore | None = None,
    use_hybrid: bool | None = None,
    reranker_kind: str | None = None,
) -> RetrievalResult:
    """End-to-end retrieval. Caller may pre-supply q_vec (e.g. from cache check)."""
    do_hybrid = settings.use_hybrid if use_hybrid is None else use_hybrid
    embedder = embedder or OpenAIEmbedder()
    store = store or VectorStore()

    if q_vec is None:
        with stage_timer(timings, "embed_ms"):
            q_vec = embedder.embed([question])[0]

    with stage_timer(timings, "retrieve_ms"):
        dense = store.query(q_vec, top_k=settings.top_k)
        top_dense = dense[0].score if dense else 0.0
        if do_hybrid:
            sparse = bm25_mod.get_index().query(question, top_k=settings.bm25_top_k)
            candidates = rrf_fuse(dense, sparse)[: settings.top_k]
        else:
            candidates = dense

    # Reranker selection.
    if settings.use_reranker:
        kind = reranker_kind or settings.reranker_kind
    else:
        kind = "none"
    rr = get_reranker(kind)
    if rr is not None and candidates:
        with stage_timer(timings, "rerank_ms"):
            candidates = rr.rerank(question, candidates, top_n=settings.rerank_top_n)
    else:
        candidates = candidates[: settings.rerank_top_n]

    return RetrievalResult(candidates=candidates, top_dense_score=top_dense)
