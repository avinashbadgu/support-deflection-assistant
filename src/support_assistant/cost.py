"""Per-call cost accounting in USD.

Prices are point-in-time and configurable. Wrong-price-data is a known weakness
(OpenAI changes prices) — we record token counts too so we can recompute later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .config import settings

# USD per 1M tokens. Sourced from OpenAI pricing page; update as needed.
PRICING_PER_1M = {
    "text-embedding-3-small": {"input": 0.020, "output": 0.000},
    "text-embedding-3-large": {"input": 0.130, "output": 0.000},
    "gpt-4o-mini":            {"input": 0.150, "output": 0.600},
    "gpt-4o":                 {"input": 2.500, "output": 10.000},
}


@dataclass
class Usage:
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def cost_usd(self) -> float:
        p = PRICING_PER_1M.get(self.model)
        if not p:
            return 0.0
        return (
            self.input_tokens * p["input"] / 1_000_000.0
            + self.output_tokens * p["output"] / 1_000_000.0
        )


def record(usage: Usage, tag: str) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with settings.cost_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "tag": tag,
                    "model": usage.model,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cost_usd": usage.cost_usd,
                }
            )
            + "\n"
        )
