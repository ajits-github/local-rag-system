from __future__ import annotations

from abc import ABC, abstractmethod


class Cleaner(ABC):
    @abstractmethod
    def clean(self, text: str) -> str:
        """Normalize raw loader output before chunking."""
