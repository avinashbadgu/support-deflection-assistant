"""Structured logging + per-event JSONL trace.

Every /ask call writes one line to events.jsonl with: question, retrieved chunk ids,
per-stage latency, answer, abstention flag, cost. Dashboard reads from this file.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

import structlog

from .config import settings

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)

log = structlog.get_logger("support_assistant")


@dataclass
class StageTimings:
    """Per-stage wall-clock latency in ms."""

    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    abstain_ms: float = 0.0
    generate_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return (
            self.embed_ms
            + self.retrieve_ms
            + self.rerank_ms
            + self.abstain_ms
            + self.generate_ms
        )


@dataclass
class AskEvent:
    """One row in events.jsonl. Powers the dashboard and the feedback loop."""

    event_id: str
    ts: float
    question: str
    retrieved_chunk_ids: list[str]
    answer: str
    abstained: bool
    abstain_reason: str | None
    cache_hit: bool
    timings: StageTimings
    cost_usd: float
    citations: list[dict[str, Any]] = field(default_factory=list)


@contextmanager
def stage_timer(timings: StageTimings, attr: str):
    """`with stage_timer(t, "embed_ms"):` populates t.embed_ms."""
    start = time.perf_counter()
    try:
        yield
    finally:
        setattr(timings, attr, (time.perf_counter() - start) * 1000.0)


def write_event(event: AskEvent) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    row = asdict(event)
    row["timings"] = asdict(event.timings)
    with settings.events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def new_event_id() -> str:
    return uuid.uuid4().hex[:12]
