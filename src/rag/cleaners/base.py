"""Cleaner interface: normalizes raw loader text before chunking."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Cleaner(ABC):
    """Normalizes a `RawDocument`'s text before it is passed to a chunker."""

    @abstractmethod
    def clean(self, text: str) -> str:
        """Normalize raw loader output before chunking.

        Parameters
        ----------
        text : str
            Raw text as produced by a `Loader`.

        Returns
        -------
        str
            Normalized text.
        """
