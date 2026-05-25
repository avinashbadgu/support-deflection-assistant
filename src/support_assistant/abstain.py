"""Abstention mechanism (M3).

The signature feature of this project: decide whether to answer at all.

Design — two gates:

  1. Score floor. If the top retrieved chunk's similarity is below
     settings.abstain_min_score, the corpus probably doesn't contain the answer.
     Cheap, deterministic.

  2. LLM judge. Even when retrieval scores look fine, the chunks may not
     actually contain the answer (off-topic but lexically similar). A small
     judge call asks "given these snippets, can the question be answered with
     direct support from them?" and returns yes/no + confidence.

Both gates are tunable. The right threshold is NOT picked by default — the M3
sweep in `support-assistant eval --sweep-abstain` produces the precision/recall
curve from which a real threshold is chosen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from . import cost
from .config import settings
from .store import Retrieved

JUDGE_PROMPT = """You decide whether a support assistant should answer a user's question
given the retrieved documentation snippets, or escalate to a human.

Rules:
- Answer ONLY if the snippets contain enough information to answer directly.
- If snippets are only tangentially related, off-topic, or only partially relevant, DO NOT answer.
- Better to escalate than to fabricate.

Return JSON: {"can_answer": <true|false>, "confidence": <0.0-1.0>, "reason": "<short>"}.
"""


@dataclass(frozen=True)
class AbstainDecision:
    abstain: bool
    reason: str
    judge_confidence: float | None  # None when the judge didn't run


def _format_snippets(context: list[Retrieved]) -> str:
    return "\n\n".join(f"[{i+1}] {r.title}\n{r.text[:800]}" for i, r in enumerate(context))


class LLMJudge:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.judge_model
        self._client = OpenAI(api_key=settings.openai_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def _call(self, messages):
        return self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

    def judge(self, question: str, context: list[Retrieved]) -> tuple[bool, float, str]:
        resp = self._call(
            [
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nSnippets:\n{_format_snippets(context)}"},
            ]
        )
        try:
            cost.record(
                cost.Usage(
                    model=self.model,
                    input_tokens=resp.usage.prompt_tokens,
                    output_tokens=resp.usage.completion_tokens,
                ),
                tag="abstain_judge",
            )
        except Exception:
            pass
        try:
            data = json.loads(resp.choices[0].message.content or "{}")
            return bool(data.get("can_answer", False)), float(data.get("confidence", 0.0)), str(data.get("reason", ""))
        except (json.JSONDecodeError, ValueError):
            return False, 0.0, "judge_parse_error"


def decide(
    question: str,
    context: list[Retrieved],
    judge: LLMJudge | None = None,
    min_score: float | None = None,
    use_judge: bool | None = None,
    top_score_override: float | None = None,
) -> AbstainDecision:
    """top_score_override lets the caller (e.g. hybrid retrieval) supply the
    true 'is this in corpus' signal — typically the top dense cosine — instead
    of context[0].score, which after RRF/rerank may be on a different scale."""
    floor = min_score if min_score is not None else settings.abstain_min_score
    do_judge = settings.abstain_use_judge if use_judge is None else use_judge

    if not context:
        return AbstainDecision(True, "no_context_retrieved", None)

    top_score = top_score_override if top_score_override is not None else context[0].score
    if top_score < floor:
        return AbstainDecision(True, f"top_score_below_floor:{top_score:.3f}<{floor}", None)

    if not do_judge:
        return AbstainDecision(False, f"floor_ok:{top_score:.3f}", None)

    judge = judge or LLMJudge()
    can_answer, confidence, reason = judge.judge(question, context)
    if not can_answer:
        return AbstainDecision(True, f"judge_no:{reason}", confidence)
    return AbstainDecision(False, f"judge_yes:{reason}", confidence)


ESCALATION_MESSAGE = (
    "I don't have a confident answer to that from the available documentation. "
    "I've flagged this for a human support agent who will follow up."
)
