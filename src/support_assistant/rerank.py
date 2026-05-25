"""Rerankers. Two implementations behind one Protocol so the eval can A/B them.

Why both:
- LLMReranker (gpt-4o-mini one-shot scorer): trivial deps, high quality on
  hard cases, slow (~1 LLM call per query), not free.
- CrossEncoderReranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`): heavy install
  (torch + transformers via sentence-transformers), but local inference is
  fast (~10ms) and free per call. This is the production-shape choice at any
  meaningful scale.

The interview-defense point is NOT "X is best" — it's that we measured both
and have an opinion grounded in the eval. See DECISIONS.md M9.
"""

from __future__ import annotations

import json
from typing import Protocol

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from . import cost
from .config import settings
from .store import Retrieved

RERANK_PROMPT = """You are a relevance judge for a support documentation system.
For each numbered snippet, rate how directly it answers the user's question on a 0-10 scale.
Return ONLY a JSON object: {"scores": [<int>, <int>, ...]} with one score per snippet, in order.
"""


class Reranker(Protocol):
    def rerank(self, question: str, candidates: list[Retrieved], top_n: int) -> list[Retrieved]: ...


class LLMReranker:
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

    def rerank(self, question: str, candidates: list[Retrieved], top_n: int) -> list[Retrieved]:
        if not candidates:
            return []
        if len(candidates) <= 1:
            return candidates[:top_n]

        snippets = "\n\n".join(
            f"[{i+1}] {c.title}\n{c.text[:800]}" for i, c in enumerate(candidates)
        )
        user = f"Question: {question}\n\nSnippets:\n{snippets}"
        resp = self._call(
            [
                {"role": "system", "content": RERANK_PROMPT},
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
                tag="rerank",
            )
        except Exception:
            pass

        try:
            data = json.loads(resp.choices[0].message.content or "{}")
            scores = data.get("scores", [])
        except (json.JSONDecodeError, AttributeError):
            scores = []

        if len(scores) != len(candidates):
            return candidates[:top_n]

        ranked = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )
        return [c for c, _ in ranked[:top_n]]


class CrossEncoderReranker:
    """Local cross-encoder via sentence-transformers. Lazy import so the
    package stays optional (only installed via the [crossencoder] extra)."""

    _model = None  # class-level cache: model load is the slow part

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.crossencoder_model
        self._load()

    def _load(self) -> None:
        if CrossEncoderReranker._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError(
                "Cross-encoder requires the optional dep. Install with:\n"
                '  uv pip install -e ".[crossencoder]"'
            ) from e
        CrossEncoderReranker._model = CrossEncoder(self.model_name)

    def rerank(self, question: str, candidates: list[Retrieved], top_n: int) -> list[Retrieved]:
        if not candidates:
            return []
        if len(candidates) <= 1:
            return candidates[:top_n]
        # Truncate each candidate to 512 chars — cross-encoder context is small,
        # longer text wastes compute without changing the ranking signal much.
        pairs = [(question, c.text[:512]) for c in candidates]
        scores = CrossEncoderReranker._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            zip(candidates, scores), key=lambda x: float(x[1]), reverse=True
        )
        return [c for c, _ in ranked[:top_n]]


def get_reranker(kind: str | None = None) -> Reranker | None:
    """Factory. Returns None for kind=='none'."""
    k = (kind or settings.reranker_kind or "llm").lower()
    if k == "none":
        return None
    if k == "llm":
        return LLMReranker()
    if k == "crossencoder":
        return CrossEncoderReranker()
    raise ValueError(f"unknown reranker_kind: {k!r}")
