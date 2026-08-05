"""Maps `ChunkingConfig.provider` to a `Chunker` instance.

Simple enough to live as an inline dispatch rather than going through
`factory.py` (see CLAUDE.md's swap-point convention).
"""

from __future__ import annotations

from rag.chunkers.base import Chunker
from rag.chunkers.recursive_chunker import RecursiveCharacterChunker
from rag.config import ChunkingConfig


def get_chunker(config: ChunkingConfig) -> Chunker:
    """Construct the `Chunker` selected by `config.provider`.

    Parameters
    ----------
    config : ChunkingConfig
        Chunking configuration.

    Returns
    -------
    Chunker
        The constructed chunker instance.

    Raises
    ------
    ValueError
        If `config.provider` names an unknown provider.
    """
    if config.provider == "recursive_character":
        return RecursiveCharacterChunker(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )
    raise ValueError(f"Unknown chunking provider: {config.provider}")
