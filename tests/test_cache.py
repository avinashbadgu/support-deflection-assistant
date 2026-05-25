"""Semantic cache: similarity threshold + TTL."""

from __future__ import annotations

import time

import pytest

from support_assistant import cache as cache_mod


@pytest.fixture
def fresh_cache(monkeypatch):
    c = cache_mod.SemanticCache()
    yield c


def test_exact_embedding_hits(fresh_cache, monkeypatch):
    monkeypatch.setattr(cache_mod.settings, "cache_similarity", 0.97)
    monkeypatch.setattr(cache_mod.settings, "cache_ttl_seconds", 60)
    e = [1.0, 0.0, 0.0]
    fresh_cache.store(e, {"answer": "x"})
    assert fresh_cache.lookup(e) == {"answer": "x"}
    assert fresh_cache.stats()["hits"] == 1


def test_orthogonal_embedding_misses(fresh_cache, monkeypatch):
    monkeypatch.setattr(cache_mod.settings, "cache_similarity", 0.97)
    monkeypatch.setattr(cache_mod.settings, "cache_ttl_seconds", 60)
    fresh_cache.store([1.0, 0.0, 0.0], {"answer": "x"})
    assert fresh_cache.lookup([0.0, 1.0, 0.0]) is None
    assert fresh_cache.stats()["misses"] == 1


def test_ttl_expires(fresh_cache, monkeypatch):
    monkeypatch.setattr(cache_mod.settings, "cache_similarity", 0.50)
    monkeypatch.setattr(cache_mod.settings, "cache_ttl_seconds", 0)  # immediate expiry
    e = [1.0, 0.0, 0.0]
    fresh_cache.store(e, {"answer": "x"})
    # Wait a tick so now > stored ts.
    time.sleep(0.01)
    assert fresh_cache.lookup(e) is None


def test_cosine_threshold_just_below(fresh_cache, monkeypatch):
    monkeypatch.setattr(cache_mod.settings, "cache_similarity", 0.999)
    monkeypatch.setattr(cache_mod.settings, "cache_ttl_seconds", 60)
    fresh_cache.store([1.0, 0.0], {"answer": "x"})
    # Slight rotation reduces cosine below threshold.
    assert fresh_cache.lookup([0.99, 0.14]) is None
