"""Fixed-size character chunking with overlap.

This is the PLACEHOLDER baseline. The eval harness (M2) compares it against
alternatives; the M4 before/after may swap this for a better strategy.
Choice of strategy must always be evidence-driven.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import settings
from .ingest import Document


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    url: str
    title: str
    idx: int
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}:{self.idx}"


def chunk_document(
    doc: Document,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    size = chunk_size or settings.chunk_size
    over = overlap if overlap is not None else settings.chunk_overlap
    if over >= size:
        raise ValueError("overlap must be smaller than chunk size")

    text = doc.text
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    step = size - over
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(doc_id=doc.doc_id, url=doc.url, title=doc.title, idx=idx, text=piece)
            )
            idx += 1
        if end == len(text):
            break
        start += step
    return chunks


def chunk_corpus(docs: list[Document]) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d))
    return out
