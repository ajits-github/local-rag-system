"""Embedder interface: turns text into embedding vectors."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Turns chunk/query text into embedding vectors."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunk texts for storage.

        Parameters
        ----------
        texts : list[str]
            Chunk texts to embed.

        Returns
        -------
        list[list[float]]
            One embedding vector per input text, same order.
        """

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for retrieval.

        Parameters
        ----------
        text : str
            Query text to embed.

        Returns
        -------
        list[float]
            The query's embedding vector.
        """
