"""Semantic cache for question answers (M6 optimization).

Stores (question_embedding, answer_payload, ts) in memory. On lookup we
cosine-compare against stored embeddings and return a hit if similarity >=
settings.cache_similarity. Time-to-live evicts stale entries.

This is the "one deliberate optimization" the spec calls for. The interview
defense is the latency/cost delta with no quality loss — measure it in M2 with
cache on vs off.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from .config import settings


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class Entry:
    embedding: list[float]
    payload: dict[str, Any]
    ts: float


@dataclass
class SemanticCache:
    entries: list[Entry] = field(default_factory=list)
    lock: RLock = field(default_factory=RLock)
    hits: int = 0
    misses: int = 0

    def lookup(self, embedding: list[float]) -> dict[str, Any] | None:
        now = time.time()
        ttl = settings.cache_ttl_seconds
        sim_threshold = settings.cache_similarity

        with self.lock:
            # Drop expired.
            self.entries = [e for e in self.entries if now - e.ts < ttl]

            best: tuple[float, Entry] | None = None
            for e in self.entries:
                s = cosine(embedding, e.embedding)
                if best is None or s > best[0]:
                    best = (s, e)

            if best and best[0] >= sim_threshold:
                self.hits += 1
                return best[1].payload
            self.misses += 1
            return None

    def store(self, embedding: list[float], payload: dict[str, Any]) -> None:
        with self.lock:
            self.entries.append(Entry(embedding=embedding, payload=payload, ts=time.time()))

    def stats(self) -> dict[str, int]:
        with self.lock:
            return {"hits": self.hits, "misses": self.misses, "size": len(self.entries)}


# Module-level singleton. FastAPI process holds one.
cache = SemanticCache()
