from __future__ import annotations

from abc import ABC, abstractmethod


class Chunker(ABC):
    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Split cleaned text into an ordered list of chunk strings."""
