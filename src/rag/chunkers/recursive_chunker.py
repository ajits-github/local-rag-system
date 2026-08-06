"""Default `Chunker`: langchain's recursive character text splitter."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.chunkers.base import Chunker
from rag.schemas import ChunkSpan


class RecursiveCharacterChunker(Chunker):
    """Splits text on a cascade of separators to stay under a target size.

    Falls back through paragraph, line, word, and character boundaries as
    needed, with `chunk_overlap` shared between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """Construct the chunker with fixed size/overlap parameters.

        Parameters
        ----------
        chunk_size : int, optional
            Maximum characters per chunk, by default 500.
        chunk_overlap : int, optional
            Characters shared between consecutive chunks, by default 50.
        """
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split(self, text: str, source_type: str | None = None) -> list[ChunkSpan]:
        """Split `text` into chunks using the configured size/overlap.

        `source_type` is accepted for interface parity with `Chunker.split`
        but ignored — this chunker has no structural awareness of any
        source type.

        Parameters
        ----------
        text : str
            Cleaned document text.
        source_type : str | None, optional
            Unused, by default None.

        Returns
        -------
        list[ChunkSpan]
            Chunk spans (no structural hints set), in document order.
        """
        return [ChunkSpan(text=t) for t in self._splitter.split_text(text)]
