"""Chunker interface: splits cleaned text into an ordered list of chunk spans."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.schemas import ChunkSpan


class Chunker(ABC):
    """Splits a `Cleaner`'s output into the chunk spans that get embedded."""

    @abstractmethod
    def split(self, text: str, source_type: str | None = None) -> list[ChunkSpan]:
        """Split cleaned text into an ordered list of chunk spans.

        Parameters
        ----------
        text : str
            Cleaned document text.
        source_type : str | None, optional
            The originating `RawDocument.source_type` (e.g. `"markdown"`,
            `"text"`, `"pdf"`), by default None. Implementations that only
            understand one syntax use this to gate their structure-aware
            parsing and fall back to plain splitting for everything else;
            callers that don't have it can omit it.

        Returns
        -------
        list[ChunkSpan]
            Chunk spans, in document order.
        """
