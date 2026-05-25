"""Incremental-update logic. No network."""

from __future__ import annotations

from support_assistant.chunk import Chunk
from support_assistant.delta import compute_delta, save_hashes


def _c(doc_id: str, idx: int, text: str) -> Chunk:
    return Chunk(doc_id=doc_id, url=f"https://x/{doc_id}", title=doc_id, idx=idx, text=text)


def test_first_run_embeds_everything():
    chunks = [_c("a", 0, "alpha"), _c("a", 1, "beta")]
    plan = compute_delta(chunks)
    assert len(plan.to_embed) == 2
    assert plan.unchanged == 0
    assert plan.to_delete == []


def test_unchanged_corpus_embeds_nothing():
    chunks = [_c("a", 0, "alpha"), _c("a", 1, "beta")]
    plan = compute_delta(chunks)
    save_hashes(plan.next_hashes)

    plan2 = compute_delta(chunks)
    assert plan2.to_embed == []
    assert plan2.unchanged == 2
    assert plan2.to_delete == []


def test_changed_chunk_is_reembedded():
    chunks = [_c("a", 0, "alpha"), _c("a", 1, "beta")]
    save_hashes(compute_delta(chunks).next_hashes)

    chunks2 = [_c("a", 0, "alpha"), _c("a", 1, "BETA-CHANGED")]
    plan = compute_delta(chunks2)
    assert len(plan.to_embed) == 1
    assert plan.to_embed[0].idx == 1
    assert plan.unchanged == 1


def test_removed_chunk_is_deleted():
    chunks = [_c("a", 0, "alpha"), _c("a", 1, "beta")]
    save_hashes(compute_delta(chunks).next_hashes)

    chunks2 = [_c("a", 0, "alpha")]
    plan = compute_delta(chunks2)
    assert plan.to_delete == ["a:1"]
    assert plan.unchanged == 1
    assert plan.to_embed == []
