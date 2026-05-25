"""Embedding provider behind a thin Protocol so models stay swappable.

Embedding calls also record usage via cost.record so the dashboard can show
spend.
"""

from __future__ import annotations

from typing import Protocol

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from . import cost
from .config import settings


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, model: str | None = None, batch_size: int = 100) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
        self.model = model or settings.embed_model
        self.batch_size = batch_size
        self._client = OpenAI(api_key=settings.openai_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def _call(self, batch: list[str]):
        return self._client.embeddings.create(model=self.model, input=batch)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="embed", leave=False):
            batch = texts[i : i + self.batch_size]
            resp = self._call(batch)
            vectors.extend(d.embedding for d in resp.data)
            try:
                cost.record(
                    cost.Usage(
                        model=self.model,
                        input_tokens=resp.usage.total_tokens,
                        output_tokens=0,
                    ),
                    tag="embed",
                )
            except Exception:
                pass
        return vectors
