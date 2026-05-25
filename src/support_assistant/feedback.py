"""Closed feedback loop: flagged answers append to feedback.jsonl.

The eval set can then ingest these as regression cases. This is the
"production failures become regression tests" loop the spec calls a
seniority signal.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

from .config import settings


@dataclass
class Feedback:
    event_id: str
    ts: float
    rating: str               # "good" | "bad"
    note: str
    question: str
    answer: str
    citations: list[dict]


def submit(event_id: str, rating: str, note: str, question: str, answer: str, citations: list[dict]) -> None:
    if rating not in ("good", "bad"):
        raise ValueError("rating must be 'good' or 'bad'")
    fb = Feedback(
        event_id=event_id,
        ts=time.time(),
        rating=rating,
        note=note,
        question=question,
        answer=answer,
        citations=citations,
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with settings.feedback_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(fb)) + "\n")


def load_all() -> list[Feedback]:
    if not settings.feedback_path.exists():
        return []
    out: list[Feedback] = []
    with settings.feedback_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(Feedback(**d))
    return out
