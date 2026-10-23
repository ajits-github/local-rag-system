"""`POST /graph/query`: answer through `GraphRAGPipeline`, with an optional per-request strategy.

Separate from `POST /query`, which is unchanged. Returns 404 when
`config.graph.enabled` is `False`.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from rag.api.auth import VerifiedIdentity
from rag.api.deps import get_config, get_current_identity, get_graphrag_pipeline, get_rate_limiter
from rag.api.request_auth import build_authorization_context, enforce_dos_limits
from rag.config import AppConfig
from rag.graph.pipeline import GraphRAGPipeline
from rag.logging_config import get_request_id

logger = logging.getLogger(__name__)

router = APIRouter()
_limiter = get_rate_limiter()

GraphFusionStrategy = Literal[
    "vector_only", "graph_only", "vector_plus_graph", "graph_seeded_vector"
]


class GraphQueryRequest(BaseModel):
    """Request body for `POST /graph/query`.

    Identity precedence matches `POST /query`: a verified JWT overrides
    `tenant_id` and `roles` from the body. `strategy`, when set, overrides
    `config.graph.fusion.strategy` for this request only.
    """

    query: str
    filters: dict[str, Any] | None = None
    strategy: GraphFusionStrategy | None = None
    tenant_id: str | None = None
    roles: list[str] | None = None
    as_of: date | None = None
    require_trust_level: str | None = None


class GraphQueryResponse(BaseModel):
    """Response body for `POST /graph/query`.

    Same shape as `QueryResponse`, plus `strategy_used`. That field is the
    requested or configured strategy, not always the one that ran: a missing
    `dataset_id` falls back to vector-only.
    """

    answer: str
    sources: list[dict[str, Any]]
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    strategy_used: str
    request_id: str | None = None


def _query_rate_limit_string() -> str:
    """Return the current `requests_per_minute` config value as a slowapi limit string."""
    return f"{get_config().security.rate_limit.requests_per_minute}/minute"


@router.post("/graph/query", response_model=GraphQueryResponse)
@_limiter.limit(_query_rate_limit_string)
def graph_query(
    request: Request,
    body: GraphQueryRequest,
    identity: VerifiedIdentity | None = Depends(get_current_identity),
    pipeline: GraphRAGPipeline = Depends(get_graphrag_pipeline),
    config: AppConfig = Depends(get_config),
) -> GraphQueryResponse:
    """Run `body.query` through `GraphRAGPipeline` and return the answer.

    Parameters
    ----------
    request : Request
        The raw HTTP request, required by `slowapi`'s rate-limit
        decorator and passed to `get_current_identity`.
    body : GraphQueryRequest
        The query, optional filters, and optional strategy override.
    identity : VerifiedIdentity | None
        The verified caller identity, or `None` when JWT auth is disabled.
    pipeline : GraphRAGPipeline
        Injected `GraphRAGPipeline` singleton.
    config : AppConfig
        Application configuration.

    Returns
    -------
    GraphQueryResponse
        The generated answer, its sources, stage timings, and which
        fusion strategy actually ran.

    Raises
    ------
    HTTPException
        404 when `config.graph.enabled` is `False`.
    """
    if not config.graph.enabled:
        raise HTTPException(status_code=404, detail="Not Found")
    enforce_dos_limits(body.query, None, body.filters, config)
    auth = build_authorization_context(
        identity, body.tenant_id, body.roles, body.as_of, body.require_trust_level
    )
    strategy_used = body.strategy or config.graph.fusion.strategy
    result = pipeline.answer(
        body.query, filters=body.filters, auth=auth, strategy_override=body.strategy
    )
    logger.info(
        "agent_request_completed",
        extra={
            "route": "graph_rag",
            "termination_reason": "synthesized",
            "step_count": 1,
            "tool_call_count": 0,
            "retrieval_attempts": None,
            "evidence_sufficient": None,
            "duration_ms": round(result["total_ms"], 2),
            "success": True,
        },
    )
    return GraphQueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        retrieval_ms=result["retrieval_ms"],
        generation_ms=result["generation_ms"],
        total_ms=result["total_ms"],
        strategy_used=strategy_used,
        request_id=get_request_id(),
    )
