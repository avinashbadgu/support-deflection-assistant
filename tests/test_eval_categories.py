"""The categorized eval set has the right structure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from support_assistant.eval.dataset import CATEGORIES, EvalCase, load_eval_set


def test_eval_set_loads_at_least_100_cases():
    cases = load_eval_set(include_feedback=False)
    assert len(cases) >= 100, f"expected 100+ cases, got {len(cases)}"


def test_every_case_has_a_known_category():
    cases = load_eval_set(include_feedback=False)
    for c in cases:
        assert c.category in CATEGORIES, f"unknown category {c.category} on {c.id}"


def test_each_required_category_is_populated():
    cases = load_eval_set(include_feedback=False)
    counts = {cat: 0 for cat in CATEGORIES}
    for c in cases:
        counts[c.category] += 1
    # Hard floors — the categories the user asked for must each have meaningful representation.
    assert counts["direct_retrieval"]  >= 25
    assert counts["exact_api"]         >= 15
    assert counts["multi_hop"]         >= 10
    assert counts["ambiguous"]         >= 10
    assert counts["adversarial"]       >= 8
    assert counts["out_of_domain"]     >= 12


def test_out_of_domain_cases_all_expect_abstain():
    for c in load_eval_set(include_feedback=False):
        if c.category == "out_of_domain":
            assert c.expected_abstain is True, f"{c.id} should expect abstain"
            assert c.expected_source_urls == [], f"{c.id} should have no expected source URLs"


def test_direct_retrieval_cases_all_have_expected_sources():
    for c in load_eval_set(include_feedback=False):
        if c.category == "direct_retrieval":
            assert c.expected_source_urls, f"{c.id} has no expected source URLs"
            assert c.expected_abstain is False


def test_expected_source_urls_are_within_indexed_corpus():
    """No expected URL should reference a page we don't ingest — that would
    make the case unrunnable."""
    from support_assistant.ingest import SEED_URLS
    allowed = {u.rstrip("/") for u in SEED_URLS}
    for c in load_eval_set(include_feedback=False):
        for u in c.expected_source_urls:
            assert u.rstrip("/") in allowed, f"{c.id} expects {u} which isn't in SEED_URLS"


def test_case_ids_are_unique():
    cases = load_eval_set(include_feedback=False)
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids detected"
