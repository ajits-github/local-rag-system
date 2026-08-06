from __future__ import annotations

from rag.chunkers.recursive_chunker import RecursiveCharacterChunker
from rag.chunkers.registry import get_chunker
from rag.chunkers.structured_markdown import StructuredMarkdownChunker
from rag.config import ChunkingConfig


def test_recursive_chunker_splits_long_text_respecting_size():
    """Long text is split into multiple chunks near the configured size."""
    chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
    text = "word " * 200  # 1000 chars

    chunks = chunker.split(text)

    assert len(chunks) > 1
    assert all(len(c.text) <= 60 for c in chunks)  # size + small slack for split boundaries


def test_recursive_chunker_returns_single_chunk_for_short_text():
    """Text shorter than chunk_size is returned as a single chunk."""
    chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.split("short text")
    assert [c.text for c in chunks] == ["short text"]


def test_registry_builds_recursive_character_chunker():
    """get_chunker builds a RecursiveCharacterChunker for that provider."""
    config = ChunkingConfig(provider="recursive_character", chunk_size=100, chunk_overlap=20)
    chunker = get_chunker(config)
    assert isinstance(chunker, RecursiveCharacterChunker)


def test_registry_builds_structured_markdown_chunker():
    """get_chunker builds a StructuredMarkdownChunker, threading its tunables through."""
    config = ChunkingConfig(provider="structured_markdown", chunk_size=100, chunk_overlap=20)
    chunker = get_chunker(config)
    assert isinstance(chunker, StructuredMarkdownChunker)
