"""End-to-end glue: build/update index, and ask a question.

`ask()` is the single source of truth for the answer flow. The API and CLI
both call it. It owns: cache lookup, retrieval (delegated to retrieve.retrieve),
abstention, generation, event logging.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from . import abstain, bm25 as bm25_mod, cache as cache_mod
from .chunk import chunk_corpus
from .config import settings
from .delta import chunk_hash, compute_delta, save_hashes
from .embed import OpenAIEmbedder
from .generate import Answer, OpenAIGenerator
from .ingest import fetch_documents
from .observability import AskEvent, StageTimings, log, new_event_id, stage_timer, write_event
from .retrieve import retrieve
from .store import Retrieved, VectorStore


def build_index(use_cache: bool = True, use_delta: bool = True) -> dict[str, int]:
    """Fetch → chunk → embed-delta → upsert → rebuild BM25. Returns counts.

    The BM25 index is always rebuilt from the FULL current chunk set (not just
    deltas) because BM25 statistics are global. Cheap enough — pure Python over
    ~hundreds of chunks.
    """
    docs = fetch_documents(use_cache=use_cache)
    chunks = chunk_corpus(docs)
    store = VectorStore()

    if use_delta:
        plan = compute_delta(chunks)
        log.info(
            "delta",
            to_embed=len(plan.to_embed),
            to_delete=len(plan.to_delete),
            unchanged=plan.unchanged,
        )
        if plan.to_delete:
            store.delete(plan.to_delete)
        if plan.to_embed:
            embedder = OpenAIEmbedder()
            vectors = embedder.embed([c.text for c in plan.to_embed])
            store.upsert(plan.to_embed, vectors)
        save_hashes(plan.next_hashes)
        bm25_mod.rebuild(chunks)
        return {
            "total_chunks": store.count(),
            "embedded": len(plan.to_embed),
            "deleted": len(plan.to_delete),
            "unchanged": plan.unchanged,
            "bm25_size": bm25_mod.get_index().size,
        }

    # Force-rebuild path.
    embedder = OpenAIEmbedder()
    vectors = embedder.embed([c.text for c in chunks])
    store.upsert(chunks, vectors)
    save_hashes({c.chunk_id: chunk_hash(c) for c in chunks})
    bm25_mod.rebuild(chunks)
    return {
        "total_chunks": store.count(),
        "embedded": len(chunks),
        "deleted": 0,
        "unchanged": 0,
        "bm25_size": bm25_mod.get_index().size,
    }


def _citations_payload(retrieved: list[Retrieved]) -> list[dict[str, Any]]:
    return [
        {"chunk_id": r.chunk_id, "url": r.url, "title": r.title, "score": r.score}
        for r in retrieved
    ]


def ask(question: str, use_cache: bool | None = None) -> dict[str, Any]:
    """End-to-end. Returns a payload dict (also written to events.jsonl).

    Keys: answer, abstained, abstain_reason, citations, timings, cost_usd,
    cache_hit, event_id.
    """
    use_cache_flag = settings.use_cache if use_cache is None else use_cache
    timings = StageTimings()
    event_id = new_event_id()
    t0 = time.perf_counter()

    embedder = OpenAIEmbedder()
    with stage_timer(timings, "embed_ms"):
        q_vec = embedder.embed([question])[0]

    if use_cache_flag:
        hit = cache_mod.cache.lookup(q_vec)
        if hit is not None:
            payload = {
                **hit,
                "cache_hit": True,
                "event_id": event_id,
                "timings": asdict(timings),
            }
            write_event(
                AskEvent(
                    event_id=event_id,
                    ts=time.time(),
                    question=question,
                    retrieved_chunk_ids=[c["chunk_id"] for c in hit.get("citations", [])],
                    answer=hit.get("answer", ""),
                    abstained=hit.get("abstained", False),
                    abstain_reason=hit.get("abstain_reason"),
                    cache_hit=True,
                    timings=timings,
                    cost_usd=0.0,
                    citations=hit.get("citations", []),
                )
            )
            return payload

    # Cache miss — delegate retrieval (we already embedded, pass q_vec through).
    result = retrieve(question, timings, q_vec=q_vec, embedder=embedder)
    candidates = result.candidates

    with stage_timer(timings, "abstain_ms"):
        # Always feed abstention the top *dense* cosine score (not RRF or rerank score)
        # so the floor threshold remains meaningful across retrieval modes.
        decision = abstain.decide(question, candidates, top_score_override=result.top_dense_score)

    if decision.abstain:
        answer_text = abstain.ESCALATION_MESSAGE
        citations = _citations_payload(candidates)
        payload: dict[str, Any] = {
            "answer": answer_text,
            "abstained": True,
            "abstain_reason": decision.reason,
            "citations": citations,
            "cache_hit": False,
            "event_id": event_id,
            "timings": asdict(timings),
        }
    else:
        with stage_timer(timings, "generate_ms"):
            gen = OpenAIGenerator()
            ans: Answer = gen.answer(question, candidates)
        citations = _citations_payload(candidates)
        payload = {
            "answer": ans.text,
            "abstained": False,
            "abstain_reason": decision.reason,
            "citations": citations,
            "cache_hit": False,
            "event_id": event_id,
            "timings": asdict(timings),
        }

    payload["cost_usd"] = _tail_cost_since(t0)

    if use_cache_flag and not payload["abstained"]:
        cache_mod.cache.store(q_vec, payload)

    write_event(
        AskEvent(
            event_id=event_id,
            ts=time.time(),
            question=question,
            retrieved_chunk_ids=[c["chunk_id"] for c in citations],
            answer=payload["answer"],
            abstained=payload["abstained"],
            abstain_reason=payload["abstain_reason"],
            cache_hit=False,
            timings=timings,
            cost_usd=payload["cost_usd"],
            citations=citations,
        )
    )
    return payload


def _tail_cost_since(t0: float) -> float:
    """Best-effort sum of recent cost.jsonl lines. Approximate."""
    import json

    if not settings.cost_path.exists():
        return 0.0
    try:
        with settings.cost_path.open(encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return 0.0
    tail = lines[-6:]
    total = 0.0
    for line in tail:
        try:
            total += float(json.loads(line).get("cost_usd", 0.0))
        except Exception:
            continue
    return total
