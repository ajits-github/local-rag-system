"""`GET /health`: reports vectorstore/LLM connectivity."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from rag.api.deps import get_config, get_graph_client, get_llm, get_vectorstore
from rag.config import AppConfig
from rag.generation.base import LLM
from rag.graph.client import Neo4jClient
from rag.vectorstore.base import VectorStore

router = APIRouter()


@router.get("/health")
def health(
    vectorstore: VectorStore = Depends(get_vectorstore),
    llm: LLM = Depends(get_llm),
    graph_client: Neo4jClient | None = Depends(get_graph_client),
    config: AppConfig = Depends(get_config),
) -> dict[str, Any]:
    """Report overall status plus each dependency's individual reachability.

    Graph is a soft dependency, like Ollama, and is probed through the
    `Neo4jClient` singleton alone, never through `GraphRAGPipeline`, so a
    cold `/health` never builds the retrieval pipeline or loads the
    embedding model. With `graph.enabled=False` there is no client and the
    graph reads as `"disabled"`, not degraded.

    Parameters
    ----------
    vectorstore : VectorStore
        Injected vector store singleton.
    llm : LLM
        Injected LLM singleton.
    graph_client : Neo4jClient | None
        Injected `Neo4jClient` singleton, `None` when graph is disabled.
    config : AppConfig
        Application configuration.

    Returns
    -------
    dict[str, Any]
        ``{"status": "ok" | "degraded", "dependencies": {...}}``.
    """
    db_ok = vectorstore.health_check()
    llm_ok = llm.health_check()
    graph_ok = graph_client is None or graph_client.health_check()
    return {
        "status": "ok" if (db_ok and llm_ok and graph_ok) else "degraded",
        "dependencies": {
            "vectorstore": "ok" if db_ok else "unreachable",
            "llm": "ok" if llm_ok else "unreachable",
            "graph": ("ok" if graph_ok else "unreachable") if config.graph.enabled else "disabled",
        },
    }
