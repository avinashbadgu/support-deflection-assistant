"""Reciprocal Rank Fusion."""

from __future__ import annotations

from support_assistant.retrieve import rrf_fuse
from support_assistant.store import Retrieved


def _r(cid: str, score: float = 0.5) -> Retrieved:
    return Retrieved(chunk_id=cid, url=f"https://x/{cid}", title="t", text="...", score=score)


def test_rrf_top_of_both_lists_wins():
    dense = [_r("a"), _r("b"), _r("c")]
    sparse = [_r("a"), _r("x"), _r("y")]
    out = rrf_fuse(dense, sparse, k=60)
    assert out[0].chunk_id == "a"   # appears at rank 0 in BOTH


def test_rrf_dedupes():
    dense = [_r("a"), _r("b")]
    sparse = [_r("b"), _r("a")]
    out = rrf_fuse(dense, sparse, k=60)
    ids = [r.chunk_id for r in out]
    assert sorted(ids) == ["a", "b"]
    assert len(ids) == 2


def test_rrf_unique_to_one_list_still_appears():
    dense = [_r("a"), _r("b")]
    sparse = [_r("c")]
    out = rrf_fuse(dense, sparse, k=60)
    ids = {r.chunk_id for r in out}
    assert ids == {"a", "b", "c"}


def test_rrf_score_is_sum_of_reciprocal_ranks():
    # k=60, item "a" at rank 0 in both → 1/61 + 1/61 = 2/61
    dense = [_r("a"), _r("b")]
    sparse = [_r("a"), _r("c")]
    out = rrf_fuse(dense, sparse, k=60)
    a = next(r for r in out if r.chunk_id == "a")
    assert abs(a.score - 2.0 / 61.0) < 1e-9


def test_rrf_preserves_metadata_from_first_seen():
    dense = [Retrieved(chunk_id="a", url="dense", title="d", text="dense-text", score=1.0)]
    sparse = [Retrieved(chunk_id="a", url="sparse", title="s", text="sparse-text", score=1.0)]
    out = rrf_fuse(dense, sparse)
    assert out[0].url == "dense"
    assert out[0].text == "dense-text"


def test_rrf_empty_inputs():
    assert rrf_fuse([], []) == []
    out = rrf_fuse([], [_r("a")])
    assert len(out) == 1 and out[0].chunk_id == "a"
