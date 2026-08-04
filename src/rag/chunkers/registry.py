from __future__ import annotations

from rag.chunkers.base import Chunker
from rag.chunkers.recursive_chunker import RecursiveCharacterChunker
from rag.config import ChunkingConfig


def get_chunker(config: ChunkingConfig) -> Chunker:
    if config.provider == "recursive_character":
        return RecursiveCharacterChunker(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )
    raise ValueError(f"Unknown chunking provider: {config.provider}")
