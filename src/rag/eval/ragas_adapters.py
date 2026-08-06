"""Bridges this project's `LLM`/`Embedder` ABCs into `langchain_core`, for RAGAS.

RAGAS's `LangchainLLMWrapper`/`LangchainEmbeddingsWrapper` need a
`langchain_core`-compatible object. Rather than adding
`langchain-openai`/`langchain-anthropic`/`langchain-ollama`, these two
thin adapters wrap this project's own `LLM`/`Embedder` ABCs directly.
`langchain_core` is already an available dependency (transitively via
`langchain`/`langchain-community`, used today by the recursive-character
chunker) — nothing new is required for this module itself; only its
caller (`ragas_scorer.py`) needs the optional `ragas` extra.
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings as LangchainEmbeddingsBase
from langchain_core.language_models.llms import LLM as LangchainLLMBase
from pydantic import ConfigDict

from rag.embedders.base import Embedder
from rag.generation.base import LLM


class LangchainLLMAdapter(LangchainLLMBase):
    """Adapts this project's `LLM` ABC to `langchain_core`'s sync LLM interface."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rag_llm: LLM

    @property
    def _llm_type(self) -> str:
        """Identify this adapter to langchain_core's internal bookkeeping."""
        return "rag-project-llm-adapter"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        """Delegate to `self.rag_llm.generate`; `stop`/`run_manager` are unsupported and ignored."""
        return self.rag_llm.generate(prompt)


class LangchainEmbeddingsAdapter(LangchainEmbeddingsBase):
    """Adapts this project's `Embedder` ABC to `langchain_core`'s Embeddings interface."""

    def __init__(self, embedder: Embedder) -> None:
        """Store the wrapped `Embedder` instance."""
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Delegate to `self._embedder.embed_documents`."""
        return self._embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Delegate to `self._embedder.embed_query`."""
        return self._embedder.embed_query(text)
