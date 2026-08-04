from __future__ import annotations

from rag.chunkers.recursive_chunker import RecursiveCharacterChunker
from rag.chunkers.registry import get_chunker
from rag.config import ChunkingConfig


def test_recursive_chunker_splits_long_text_respecting_size():
    chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
    text = "word " * 200  # 1000 chars

    chunks = chunker.split(text)

    assert len(chunks) > 1
    assert all(len(c) <= 60 for c in chunks)  # size + small slack for split boundaries


def test_recursive_chunker_returns_single_chunk_for_short_text():
    chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.split("short text")
    assert chunks == ["short text"]


def test_registry_builds_recursive_character_chunker():
    config = ChunkingConfig(provider="recursive_character", chunk_size=100, chunk_overlap=20)
    chunker = get_chunker(config)
    assert isinstance(chunker, RecursiveCharacterChunker)
