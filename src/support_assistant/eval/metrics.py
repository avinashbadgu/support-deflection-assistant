"""Pure metric functions. NO LLM calls here — those live in judge.py.

Keeping the math pure means tests/test_metrics.py can verify correctness
deterministically. CLAUDE.md is explicit: the metrics must be trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass


def recall_at_k(expected_urls: list[str], retrieved_urls: list[str]) -> float:
    """Did we retrieve at least one expected source? 1.0 or 0.0.

    Any-of semantics: hitting any one of the expected URLs counts. Cases
    with no expected source (abstain cases) return 1.0 by convention —
    they're scored on abstention accuracy instead.
    """
    if not expected_urls:
        return 1.0
    expected = {u.rstrip("/") for u in expected_urls}
    seen = {u.rstrip("/") for u in retrieved_urls}
    return 1.0 if expected & seen else 0.0


@dataclass
class AbstentionCounts:
    tp: int = 0   # correctly abstained when answer not in corpus
    tn: int = 0   # correctly answered when answer in corpus
    fp: int = 0   # abstained when it should have answered (missed answer)
    fn: int = 0   # answered when it should have abstained (hallucination risk)

    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total else 0.0

    @property
    def precision_abstain(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall_abstain(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0


def update_abstention(counts: AbstentionCounts, expected_abstain: bool, did_abstain: bool) -> None:
    if expected_abstain and did_abstain:
        counts.tp += 1
    elif not expected_abstain and not did_abstain:
        counts.tn += 1
    elif not expected_abstain and did_abstain:
        counts.fp += 1
    else:
        counts.fn += 1


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile. p in [0,1]. Empty -> 0.0."""
    if not values:
        return 0.0
    s = sorted(values)
    if p <= 0:
        return s[0]
    if p >= 1:
        return s[-1]
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


@dataclass
class CategoryBucket:
    category: str
    n: int = 0
    recalls: list[float] = None
    abstention: AbstentionCounts = None
    faithfulnesses: list[float] = None
    latencies_ms: list[float] = None

    def __post_init__(self):
        if self.recalls is None:
            self.recalls = []
        if self.abstention is None:
            self.abstention = AbstentionCounts()
        if self.faithfulnesses is None:
            self.faithfulnesses = []
        if self.latencies_ms is None:
            self.latencies_ms = []

    def summary(self) -> dict:
        return {
            "category": self.category,
            "n": self.n,
            "recall_at_k_mean": (sum(self.recalls) / len(self.recalls)) if self.recalls else None,
            "abstention_accuracy": self.abstention.accuracy if (self.abstention.tp + self.abstention.tn + self.abstention.fp + self.abstention.fn) else None,
            "faithfulness_mean": (sum(self.faithfulnesses) / len(self.faithfulnesses)) if self.faithfulnesses else None,
            "latency_p95_ms": percentile(self.latencies_ms, 0.95),
        }
