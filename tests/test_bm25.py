"""BM25 sparse retrieval — tokenizer + scoring."""

from __future__ import annotations

import pytest

from support_assistant.bm25 import BM25Index, tokenize
from support_assistant.chunk import Chunk


def _c(idx: int, text: str) -> Chunk:
    return Chunk(doc_id="d", url=f"https://x/{idx}", title="t", idx=idx, text=text)


def test_tokenize_basic():
    toks = tokenize("Hello World — refund_amount = $42!")
    assert "hello" in toks
    assert "world" in toks
    assert "refund_amount" in toks
    assert "42" in toks
    assert "$" not in toks
    assert "—" not in toks


def test_tokenize_drops_short_tokens():
    toks = tokenize("a I is bb cc")
    assert "is" in toks   # 2 chars passes
    assert "bb" in toks
    assert "cc" in toks
    assert "a" not in toks  # 1 char dropped
    assert "i" not in toks


def test_bm25_finds_keyword_match():
    idx = BM25Index()
    idx.build([
        _c(0, "Refunds are processed via the refund endpoint."),
        _c(1, "Subscriptions bill on a recurring schedule."),
        _c(2, "Webhook signatures use HMAC-SHA256."),
    ])
    out = idx.query("how do refunds work", top_k=2)
    assert len(out) >= 1
    # The refunds chunk should be the top hit.
    assert "refund" in out[0].text.lower()


def test_bm25_empty_query_returns_nothing():
    idx = BM25Index()
    idx.build([_c(0, "alpha beta gamma")])
    assert idx.query("", top_k=5) == []
    assert idx.query("!!!", top_k=5) == []


def test_bm25_persists_and_reloads():
    # Need enough docs that BM25 IDF doesn't collapse to zero — with only 2 docs,
    # any term in exactly 1 doc has IDF = log((2 - 1 + 0.5) / (1 + 0.5)) = log(1) = 0.
    idx = BM25Index()
    idx.build([
        _c(0, "refund partial amount on a charge"),
        _c(1, "webhook signature timestamp verification"),
        _c(2, "subscription billing schedule and trial periods"),
        _c(3, "connect onboarding for marketplace platforms"),
        _c(4, "stripe radar fraud rules and review"),
    ])
    out_first = idx.query("refund amount", top_k=1)
    assert out_first, "expected at least one match for 'refund amount'"

    idx2 = BM25Index()
    assert idx2.load() is True
    out_second = idx2.query("refund amount", top_k=1)
    assert out_first[0].chunk_id == out_second[0].chunk_id


def test_bm25_load_returns_false_when_absent(tmp_path, monkeypatch):
    from support_assistant.config import settings

    # tmp_path fixture is autouse'd in conftest — settings already redirected.
    idx = BM25Index()
    assert idx.load() is False
