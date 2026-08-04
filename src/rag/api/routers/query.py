from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from rag.api.deps import get_retrieval_pipeline
from rag.retrieval.pipeline import RetrievalPipeline

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None
    filters: dict[str, Any] | None = None


class SourceItem(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
) -> QueryResponse:
    result = pipeline.answer(request.query, filters=request.filters, top_k=request.top_k)
    return QueryResponse(**result)
