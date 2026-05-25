"""Chroma wrapper. Thin so we can migrate to pgvector/Qdrant later."""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from .chunk import Chunk
from .config import settings

COLLECTION = "stripe_docs"


@dataclass(frozen=True)
class Retrieved:
    chunk_id: str
    url: str
    title: str
    text: str
    score: float  # similarity (1 - cosine distance); higher is better


class VectorStore:
    def __init__(self) -> None:
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {"url": c.url, "title": c.title, "doc_id": c.doc_id, "idx": c.idx}
                for c in chunks
            ],
        )

    def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        self._collection.delete(ids=chunk_ids)

    def query(self, embedding: list[float], top_k: int | None = None) -> list[Retrieved]:
        k = top_k or settings.top_k
        res = self._collection.query(query_embeddings=[embedding], n_results=k)
        out: list[Retrieved] = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
            # Chroma returns cosine distance in [0, 2]; convert to similarity.
            sim = 1.0 - float(dist)
            out.append(
                Retrieved(
                    chunk_id=chunk_id,
                    url=meta.get("url", ""),
                    title=meta.get("title", ""),
                    text=text,
                    score=sim,
                )
            )
        return out

    def count(self) -> int:
        return self._collection.count()
