"""`POST /query`: answer a question via `RetrievalPipeline.answer`."""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from rag.api.deps import get_retrieval_pipeline
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.pipeline import RetrievalPipeline

router = APIRouter()


class QueryRequest(BaseModel):
    """Request body for `POST /query`.

    `tenant_id`/`roles`/`as_of` are trusted caller-identity claims (see
    `retrieval/authorization.py`'s docstring for why: no authentication/
    session layer exists in this codebase, so these are asserted directly
    by the caller rather than verified here -- a real deployment would
    populate them from a verified JWT/session claim at this same boundary
    instead of a raw request field). All optional and `None` by default,
    matching `RetrievalPipeline`'s "`None` = fully unrestricted" semantics,
    and inert unless `config.security.authorization.enabled` is also `True`.
    """

    query: str
    top_k: int | None = None
    filters: dict[str, Any] | None = None
    tenant_id: str | None = None
    roles: list[str] | None = None
    as_of: date | None = None
    require_trust_level: str | None = None


class SourceItem(BaseModel):
    """One retrieved-chunk citation in a `QueryResponse`."""

    chunk_id: str
    document_id: str
    source: str
    category: str | None = None
    score: float


class QueryResponse(BaseModel):
    """Response body for `POST /query`."""

    answer: str
    sources: list[SourceItem]
    retrieval_ms: float
    generation_ms: float
    total_ms: float


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
) -> QueryResponse:
    """Run `request.query` through the retrieval pipeline and return the answer.

    Parameters
    ----------
    request : QueryRequest
        The query, plus optional `top_k`/`filters` overrides. `top_k`
        maps onto `RetrievalPipeline.answer`'s `candidate_k` (the
        external API field name is kept stable; only the internal
        parameter name changed -- see `retrieval/pipeline.py`).
    pipeline : RetrievalPipeline
        Injected retrieval pipeline singleton.

    Returns
    -------
    QueryResponse
        The generated answer, its sources, and stage timings.
    """
    has_auth_fields = (
        request.tenant_id is not None
        or request.roles
        or request.as_of is not None
        or request.require_trust_level is not None
    )
    auth = (
        AuthorizationContext(
            tenant_id=request.tenant_id,
            roles=request.roles or [],
            as_of=request.as_of,
            require_trust_level=request.require_trust_level,
        )
        if has_auth_fields
        else None
    )
    result = pipeline.answer(
        request.query, filters=request.filters, candidate_k=request.top_k, auth=auth
    )
    return QueryResponse(**result)
