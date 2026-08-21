"""`POST /agent/query`: answer a question via the bounded agentic RAG workflow.

A separate endpoint from `POST /query` (not a mode flag). See
`docs/architecture.md`'s "Agentic RAG" section for the tradeoff. Reuses,
rather than reimplements, `/query`'s exact JWT-precedence and DoS-limit
logic (`rag.api.request_auth`) and the same DI singletons
(`rag.api.deps`), so this route's identity/authorization/rate-limit/audit
behavior is byte-identical to `/query`'s.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from rag.agent.graph import run_agent
from rag.agent.state import AgentState
from rag.api.auth import VerifiedIdentity
from rag.api.deps import (
    get_config,
    get_current_identity,
    get_embedder,
    get_llm,
    get_rate_limiter,
    get_retrieval_pipeline,
    get_vectorstore,
)
from rag.api.request_auth import build_authorization_context, enforce_dos_limits
from rag.api.routers.query import SourceItem
from rag.config import AppConfig
from rag.embedders.base import Embedder
from rag.generation.base import LLM
from rag.retrieval.pipeline import RetrievalPipeline
from rag.vectorstore.base import VectorStore

router = APIRouter()
_limiter = get_rate_limiter()


class AgentQueryRequest(BaseModel):
    """Request body for `POST /agent/query`.

    Same identity-claim precedence as `QueryRequest` (see
    `rag.api.request_auth.build_authorization_context`): `tenant_id`/
    `roles` are only trusted when no verified JWT identity is present.
    """

    query: str
    filters: dict[str, Any] | None = None
    tenant_id: str | None = None
    roles: list[str] | None = None
    as_of: date | None = None
    require_trust_level: str | None = None


class AgentQueryResponse(BaseModel):
    """Response body for `POST /agent/query`."""

    answer: str
    sources: list[SourceItem]
    route: str
    termination_reason: str | None
    steps: int
    tool_calls: list[str]
    retrieval_ms: float
    generation_ms: float
    total_ms: float


def _agent_rate_limit_string() -> str:
    """Return the current `requests_per_minute` config value as a slowapi limit string.

    Reuses the same `security.rate_limit` budget as `/query` for v1 --
    see `docs/architecture.md`'s "Agentic RAG" section for why a
    shared budget was chosen over a stricter agent-specific one, and
    when to revisit that.
    """
    return f"{get_config().security.rate_limit.requests_per_minute}/minute"


@router.post("/agent/query", response_model=AgentQueryResponse)
@_limiter.limit(_agent_rate_limit_string)
def agent_query(
    request: Request,
    body: AgentQueryRequest,
    identity: VerifiedIdentity | None = Depends(get_current_identity),
    pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
    vectorstore: VectorStore = Depends(get_vectorstore),
    embedder: Embedder = Depends(get_embedder),
    llm: LLM = Depends(get_llm),
    config: AppConfig = Depends(get_config),
) -> AgentQueryResponse:
    """Run `body.query` through the bounded agent graph and return the answer.

    Parameters
    ----------
    request : Request
        The raw HTTP request, required by `slowapi`'s rate-limit
        decorator and passed to `get_current_identity`.
    body : AgentQueryRequest
        The query, plus optional `filters` and identity-claim fields.
    identity : VerifiedIdentity | None
        The verified caller identity, or `None` when JWT auth is disabled.
    pipeline, vectorstore, embedder, llm : injected singletons
        The same process-wide instances `/query` uses.
    config : AppConfig
        Application configuration; `config.agent` governs routing/bounds.

    Returns
    -------
    AgentQueryResponse
        The generated answer, its sources, and the route/step/tool-call
        summary. When `config.agent.enabled` is `False`, always takes the
        `"classic_rag"` route with zero extra LLM calls.
    """
    enforce_dos_limits(body.query, None, body.filters, config)
    auth = build_authorization_context(
        identity, body.tenant_id, body.roles, body.as_of, body.require_trust_level
    )
    state = AgentState(original_query=body.query, authorization_context=auth, filters=body.filters)
    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=vectorstore,
        embedder=embedder,
        llm=llm,
        config=config,
    )
    final_state = result.state
    return AgentQueryResponse(
        answer=final_state.final_answer or "",
        sources=[
            SourceItem(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                source=c.source,
                category=c.category,
                score=c.score if c.score is not None else 0.0,
                content_type=c.content_type,
                section_path=c.section_path,
                page=c.page,
                attachment_name=c.attachment_name,
                source_anchor=c.source_anchor,
                vision_generated=c.vision_generated,
            )
            for c in final_state.citations
        ],
        route=result.route,
        termination_reason=final_state.termination_reason,
        steps=final_state.step_count,
        tool_calls=[record.tool_name for record in final_state.tool_call_history],
        retrieval_ms=result.retrieval_ms,
        generation_ms=result.generation_ms,
        total_ms=result.total_ms,
    )
