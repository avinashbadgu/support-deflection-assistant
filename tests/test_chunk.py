"""Chunking math. Pure — no API calls."""

from __future__ import annotations

from support_assistant.chunk import chunk_document
from support_assistant.ingest import Document


def _doc(text: str) -> Document:
    return Document(url="https://x/y", title="t", text=text)


def test_short_doc_one_chunk():
    chunks = chunk_document(_doc("hello world"), chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


def test_chunks_cover_text_with_overlap():
    text = "abcdefghij" * 30  # 300 chars
    chunks = chunk_document(_doc(text), chunk_size=100, overlap=20)
    # step = 80; 300 / 80 -> 4 chunks (start 0, 80, 160, 240)
    assert len(chunks) == 4
    # First chunk starts at 0
    assert chunks[0].text.startswith(text[:10])
    # Last chunk ends at end of text
    assert chunks[-1].text.endswith(text[-10:])


def test_overlap_must_be_smaller_than_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_document(_doc("x" * 200), chunk_size=100, overlap=100)


def test_chunk_ids_are_unique_and_ordered():
    chunks = chunk_document(_doc("z" * 1000), chunk_size=300, overlap=50)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    indices = [c.idx for c in chunks]
    assert indices == sorted(indices)
