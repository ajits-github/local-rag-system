"""`POST /query`: answer a question via `RetrievalPipeline.answer`."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from rag.api.deps import get_retrieval_pipeline
from rag.retrieval.pipeline import RetrievalPipeline

router = APIRouter()


class QueryRequest(BaseModel):
    """Request body for `POST /query`."""

    query: str
    top_k: int | None = None
    filters: dict[str, Any] | None = None


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
    result = pipeline.answer(request.query, filters=request.filters, candidate_k=request.top_k)
    return QueryResponse(**result)
