"""Eval set loader.

Cases are categorized so reports break out per-category metrics.
Categories:
  - direct_retrieval     single source, clearly answered in one chunk
  - exact_api            specific endpoint / parameter / error / identifier lookup
  - multi_hop            answer requires fusing 2+ source chunks
  - ambiguous            vague phrasing, multiple plausible interpretations
  - adversarial          lexically similar to corpus but off-topic
  - stale_conflicting    corpus may give conflicting / outdated info
  - out_of_domain        deliberately outside the corpus — must abstain
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import settings
from ..feedback import load_all as load_feedback

SEED_PATH = Path(__file__).parent / "data" / "seed_questions.json"

CATEGORIES = [
    "direct_retrieval",
    "exact_api",
    "multi_hop",
    "ambiguous",
    "adversarial",
    "stale_conflicting",
    "out_of_domain",
]


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    expected_source_urls: list[str]
    expected_abstain: bool
    gold_answer_summary: str
    category: str = "direct_retrieval"


def load_eval_set(include_feedback: bool = True) -> list[EvalCase]:
    raw = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for item in raw:
        cases.append(
            EvalCase(
                id=item["id"],
                question=item["question"],
                expected_source_urls=item.get("expected_source_urls", []),
                expected_abstain=bool(item.get("expected_abstain", False)),
                gold_answer_summary=item.get("gold_answer_summary", ""),
                category=item.get("category", "direct_retrieval"),
            )
        )

    if not include_feedback:
        return cases

    # Closed feedback loop — bad-flagged production answers become regression cases.
    for fb in load_feedback():
        if fb.rating != "bad":
            continue
        cases.append(
            EvalCase(
                id=f"fb-{fb.event_id}",
                question=fb.question,
                expected_source_urls=[c.get("url", "") for c in fb.citations if c.get("url")],
                expected_abstain=False,
                gold_answer_summary=fb.note or "(flagged; please add a gold answer)",
                category="feedback_regression",
            )
        )

    return cases


def report_dir() -> Path:
    p = settings.data_dir / "eval_reports"
    p.mkdir(parents=True, exist_ok=True)
    return p
