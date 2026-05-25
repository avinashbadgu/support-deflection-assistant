"""Incremental updates: hash chunks, persist hash map, only re-embed changes.

M5 deliverable. Re-running ingest on an unchanged corpus must do zero embedding
work. Re-running after one upstream doc changes must only re-embed that doc's chunks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .chunk import Chunk
from .config import settings


def chunk_hash(c: Chunk) -> str:
    return hashlib.sha256(c.text.encode("utf-8")).hexdigest()


def load_hashes() -> dict[str, str]:
    if not settings.hashes_path.exists():
        return {}
    return json.loads(settings.hashes_path.read_text(encoding="utf-8"))


def save_hashes(h: dict[str, str]) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.hashes_path.write_text(json.dumps(h, indent=2), encoding="utf-8")


@dataclass
class DeltaPlan:
    to_embed: list[Chunk]           # new or changed
    to_delete: list[str]            # chunk_ids that no longer exist
    unchanged: int                  # count
    next_hashes: dict[str, str]     # to persist after embed/upsert success


def compute_delta(current_chunks: list[Chunk]) -> DeltaPlan:
    """Compare current chunk set to last-known hashes; return what to embed/delete."""
    prev = load_hashes()
    cur_hashes = {c.chunk_id: chunk_hash(c) for c in current_chunks}

    to_embed: list[Chunk] = []
    unchanged = 0
    for c in current_chunks:
        h = cur_hashes[c.chunk_id]
        if prev.get(c.chunk_id) == h:
            unchanged += 1
        else:
            to_embed.append(c)

    to_delete = [cid for cid in prev.keys() if cid not in cur_hashes]
    return DeltaPlan(
        to_embed=to_embed,
        to_delete=to_delete,
        unchanged=unchanged,
        next_hashes=cur_hashes,
    )
