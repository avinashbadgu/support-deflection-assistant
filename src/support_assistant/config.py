"""Env-driven settings. One source of truth for tunable knobs.

Knobs that were picked as placeholders (chunk_size, top_k, abstain_min_score)
are flagged in DECISIONS.md and must be revisited once the eval harness
produces real numbers.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SA_",
        extra="ignore",
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    embed_model: str = "text-embedding-3-small"
    gen_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"

    data_dir: Path = Path("./data")

    # Retrieval knobs.
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 8           # retrieve this many before reranking
    rerank_top_n: int = 5    # keep this many after rerank

    # Hybrid retrieval (M8): BM25 + dense, fused with RRF.
    use_hybrid: bool = False
    bm25_top_k: int = 20
    rrf_k: int = 60

    # A/B toggles for the M4 / M9 before/after.
    use_reranker: bool = True
    reranker_kind: str = "llm"   # "llm" | "crossencoder" | "none"
    crossencoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_cache: bool = True

    # Abstention.
    abstain_min_score: float = 0.35   # cosine-similarity floor on top-1
    abstain_use_judge: bool = True

    # Semantic cache (M6 optimization).
    cache_similarity: float = 0.97
    cache_ttl_seconds: int = 3600

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def events_path(self) -> Path:
        return self.data_dir / "events.jsonl"

    @property
    def feedback_path(self) -> Path:
        return self.data_dir / "feedback.jsonl"

    @property
    def drops_path(self) -> Path:
        return self.data_dir / "drops.jsonl"

    @property
    def cost_path(self) -> Path:
        return self.data_dir / "cost.jsonl"

    @property
    def hashes_path(self) -> Path:
        return self.data_dir / "chunk_hashes.json"

    @property
    def bm25_path(self) -> Path:
        return self.data_dir / "bm25_index.json"


settings = Settings()
