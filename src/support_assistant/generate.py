"""Grounded answer generation with citations.

The answer step never runs when abstention has fired — by then the pipeline
returns the escalation message. So this module can assume context is on-topic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from . import cost
from .config import settings
from .store import Retrieved

SYSTEM_PROMPT = """You are a support assistant for Stripe.
Answer the user's question using ONLY the numbered context snippets below.
Cite the snippet numbers you used in square brackets like [1], [2].
If multiple snippets support a claim, cite all of them.
Keep the answer concise. Do not invent links, error codes, or API field names that are not in the snippets.
"""


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Retrieved]


class Generator(Protocol):
    model: str

    def answer(self, question: str, context: list[Retrieved]) -> Answer: ...


def format_context(context: list[Retrieved]) -> str:
    return "\n\n".join(
        f"[{i+1}] {r.title} ({r.url})\n{r.text}" for i, r in enumerate(context)
    )


class OpenAIGenerator:
    def __init__(self, model: str | None = None) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
        self.model = model or settings.gen_model
        self._client = OpenAI(api_key=settings.openai_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def _call(self, messages):
        return self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )

    def answer(self, question: str, context: list[Retrieved]) -> Answer:
        user_msg = f"Question: {question}\n\nContext:\n{format_context(context)}"
        resp = self._call(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
        )
        try:
            cost.record(
                cost.Usage(
                    model=self.model,
                    input_tokens=resp.usage.prompt_tokens,
                    output_tokens=resp.usage.completion_tokens,
                ),
                tag="generate",
            )
        except Exception:
            pass
        return Answer(text=resp.choices[0].message.content or "", citations=context)
