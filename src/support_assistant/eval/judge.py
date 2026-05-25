"""LLM judge for faithfulness.

Given (question, answer, citations), the judge returns:
- supported: every claim in the answer is supported by the citations
- score: 0.0-1.0 fraction of claims that are supported
- unsupported_claims: list of claims it could not find in the citations

We deliberately use the same model family as generation to keep the harness
simple. A stronger setup uses a different model for judging — this is a known
weakness and explicitly noted in DECISIONS.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .. import cost
from ..config import settings

FAITH_PROMPT = """You judge whether an assistant's answer is fully grounded in the provided source snippets.
Break the answer into discrete factual claims.
For each claim, decide if it is directly supported by at least one snippet.
Return JSON: {
  "score": <float 0.0-1.0, fraction of claims supported>,
  "supported": <true iff score == 1.0>,
  "unsupported_claims": [<string>, ...]
}
Be strict. If a claim adds detail not in any snippet, mark it unsupported.
"""


@dataclass(frozen=True)
class FaithfulnessResult:
    score: float
    supported: bool
    unsupported_claims: list[str]


class FaithfulnessJudge:
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

    def judge(self, question: str, answer: str, citations: list[dict]) -> FaithfulnessResult:
        snippets = "\n\n".join(f"[{i+1}] {c.get('title','')}\n{c.get('text','')[:1500]}" for i, c in enumerate(citations))
        user = f"Question: {question}\n\nAnswer:\n{answer}\n\nSnippets:\n{snippets}"
        resp = self._call(
            [
                {"role": "system", "content": FAITH_PROMPT},
                {"role": "user", "content": user},
            ]
        )
        try:
            cost.record(
                cost.Usage(
                    model=self.model,
                    input_tokens=resp.usage.prompt_tokens,
                    output_tokens=resp.usage.completion_tokens,
                ),
                tag="faithfulness_judge",
            )
        except Exception:
            pass
        try:
            data = json.loads(resp.choices[0].message.content or "{}")
            score = float(data.get("score", 0.0))
            return FaithfulnessResult(
                score=score,
                supported=bool(data.get("supported", score >= 0.999)),
                unsupported_claims=list(data.get("unsupported_claims", [])),
            )
        except (json.JSONDecodeError, ValueError):
            return FaithfulnessResult(0.0, False, ["judge_parse_error"])
