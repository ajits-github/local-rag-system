"""Builds pipeline objects once from config and shares them across requests.

Every getter is `lru_cache`d with no arguments, so each is a process-wide
singleton — important on an 8GB-RAM CPU-only box where we don't want two
copies of the embedding model or two separate DB connection pools.
"""

from __future__ import annotations

from functools import lru_cache

from rag.config import AppConfig, load_config
from rag.embedders.base import Embedder
from rag.factory import build_embedder, build_llm, build_reranker, build_vectorstore
from rag.generation.base import LLM
from rag.ingestion.pipeline import IngestionPipeline
from rag.rerankers.base import Reranker
from rag.retrieval.pipeline import RetrievalPipeline
from rag.vectorstore.base import VectorStore


@lru_cache
def get_config() -> AppConfig:
    return load_config()


@lru_cache
def get_embedder() -> Embedder:
    return build_embedder(get_config())


@lru_cache
def get_vectorstore() -> VectorStore:
    return build_vectorstore(get_config())


@lru_cache
def get_reranker() -> Reranker:
    return build_reranker(get_config())


@lru_cache
def get_llm() -> LLM:
    return build_llm(get_config())


@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline(get_config(), vectorstore=get_vectorstore(), embedder=get_embedder())


@lru_cache
def get_retrieval_pipeline() -> RetrievalPipeline:
    return RetrievalPipeline(
        get_config(),
        vectorstore=get_vectorstore(),
        embedder=get_embedder(),
        reranker=get_reranker(),
        llm=get_llm(),
    )
