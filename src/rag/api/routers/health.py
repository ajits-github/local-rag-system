"""`GET /health`: reports vectorstore/LLM connectivity."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from rag.api.deps import get_config, get_graphrag_pipeline, get_llm, get_vectorstore
from rag.config import AppConfig
from rag.generation.base import LLM
from rag.graph.pipeline import GraphRAGPipeline
from rag.vectorstore.base import VectorStore

router = APIRouter()


@router.get("/health")
def health(
    vectorstore: VectorStore = Depends(get_vectorstore),
    llm: LLM = Depends(get_llm),
    graph_pipeline: GraphRAGPipeline = Depends(get_graphrag_pipeline),
    config: AppConfig = Depends(get_config),
) -> dict[str, Any]:
    """Report overall status plus each dependency's individual reachability.

    Graph is a soft dependency, like Ollama. Its `health_check()` returns
    `True` when `graph.enabled=False`, so the endpoint is not degraded for a
    disabled graph.

    Parameters
    ----------
    vectorstore : VectorStore
        Injected vector store singleton.
    llm : LLM
        Injected LLM singleton.
    graph_pipeline : GraphRAGPipeline
        Injected `GraphRAGPipeline` singleton, used only for its
        `health_check()`.
    config : AppConfig
        Application configuration.

    Returns
    -------
    dict[str, Any]
        ``{"status": "ok" | "degraded", "dependencies": {...}}``.
    """
    db_ok = vectorstore.health_check()
    llm_ok = llm.health_check()
    graph_ok = graph_pipeline.health_check()
    return {
        "status": "ok" if (db_ok and llm_ok and graph_ok) else "degraded",
        "dependencies": {
            "vectorstore": "ok" if db_ok else "unreachable",
            "llm": "ok" if llm_ok else "unreachable",
            "graph": ("ok" if graph_ok else "unreachable") if config.graph.enabled else "disabled",
        },
    }
