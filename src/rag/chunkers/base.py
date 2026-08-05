"""Chunker interface: splits cleaned text into an ordered list of chunks."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Chunker(ABC):
    """Splits a `Cleaner`'s output into the chunk strings that get embedded."""

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Split cleaned text into an ordered list of chunk strings.

        Parameters
        ----------
        text : str
            Cleaned document text.

        Returns
        -------
        list[str]
            Chunk strings, in document order.
        """
