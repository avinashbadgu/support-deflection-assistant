"""API smoke tests with the real pipeline stubbed."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    # Stub pipeline.ask so we don't hit OpenAI.
    from support_assistant import pipeline

    def fake_ask(q, use_cache=None):
        return {
            "answer": "stub answer for: " + q,
            "abstained": False,
            "abstain_reason": "floor_ok:0.900",
            "citations": [{"chunk_id": "x:0", "url": "https://x", "title": "t", "score": 0.9}],
            "cache_hit": False,
            "event_id": "evt1",
            "timings": {"embed_ms": 1, "retrieve_ms": 2, "rerank_ms": 3, "abstain_ms": 4, "generate_ms": 5},
            "cost_usd": 0.001,
        }

    monkeypatch.setattr(pipeline, "ask", fake_ask)

    from support_assistant.api.app import create_app

    return TestClient(create_app())


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ask_endpoint(client):
    r = client.post("/api/ask", json={"question": "what is stripe?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("stub answer")
    assert body["abstained"] is False
    assert body["event_id"] == "evt1"


def test_ask_rejects_empty(client):
    r = client.post("/api/ask", json={"question": "   "})
    assert r.status_code == 400


def test_feedback_endpoint(client):
    r = client.post(
        "/api/feedback",
        json={
            "event_id": "evt1",
            "rating": "good",
            "note": "looks right",
            "question": "q",
            "answer": "a",
            "citations": [],
        },
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_feedback_rejects_bad_rating(client):
    r = client.post(
        "/api/feedback",
        json={
            "event_id": "evt1",
            "rating": "meh",
            "note": "",
            "question": "q",
            "answer": "a",
            "citations": [],
        },
    )
    assert r.status_code == 400


def test_metrics_empty(client):
    r = client.get("/api/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 0
