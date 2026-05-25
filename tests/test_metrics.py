"""Tests for the metric math — required by CLAUDE.md ('metrics must be trustworthy')."""

from __future__ import annotations

import math

from support_assistant.eval.metrics import (
    AbstentionCounts,
    percentile,
    recall_at_k,
    update_abstention,
)


def test_recall_at_k_hit():
    assert recall_at_k(["https://x/a"], ["https://x/a", "https://x/b"]) == 1.0


def test_recall_at_k_miss():
    assert recall_at_k(["https://x/a"], ["https://x/b", "https://x/c"]) == 0.0


def test_recall_at_k_any_of_semantics():
    # Multiple acceptable expected sources — hitting any one is a success.
    assert recall_at_k(["https://x/a", "https://x/b"], ["https://x/b"]) == 1.0


def test_recall_at_k_normalizes_trailing_slash():
    assert recall_at_k(["https://x/a/"], ["https://x/a"]) == 1.0


def test_recall_at_k_abstain_case_returns_one():
    # By convention abstain cases ignore recall.
    assert recall_at_k([], ["https://x/anything"]) == 1.0


def test_abstention_counts_perfect():
    c = AbstentionCounts()
    update_abstention(c, expected_abstain=True, did_abstain=True)
    update_abstention(c, expected_abstain=False, did_abstain=False)
    assert c.tp == 1 and c.tn == 1 and c.fp == 0 and c.fn == 0
    assert c.accuracy == 1.0


def test_abstention_counts_hallucination_case():
    # Should have abstained, but answered — fn.
    c = AbstentionCounts()
    update_abstention(c, expected_abstain=True, did_abstain=False)
    assert c.fn == 1
    assert c.recall_abstain == 0.0


def test_abstention_counts_overcautious_case():
    # Should have answered, but abstained — fp.
    c = AbstentionCounts()
    update_abstention(c, expected_abstain=False, did_abstain=True)
    assert c.fp == 1
    assert c.precision_abstain == 0.0


def test_percentile_basic():
    assert math.isclose(percentile([1, 2, 3, 4, 5], 0.0), 1.0)
    assert math.isclose(percentile([1, 2, 3, 4, 5], 1.0), 5.0)
    assert math.isclose(percentile([1, 2, 3, 4, 5], 0.5), 3.0)


def test_percentile_empty():
    assert percentile([], 0.5) == 0.0


def test_percentile_p95_interpolates():
    vals = list(range(1, 101))  # 1..100
    # p95 across 100 evenly spaced points: k = 99*0.95 = 94.05 → s[94]+(s[95]-s[94])*0.05
    # s[94]=95, s[95]=96 → 95.05
    assert math.isclose(percentile(vals, 0.95), 95.05, rel_tol=1e-6)
