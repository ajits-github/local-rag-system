"""`POST /agent/query/stream`: Server-Sent Events for live agent progress.

Additional to, never a replacement for, `POST /agent/query`
(`agent_query.py`, untouched by this module. Both routes share the same
JWT-precedence/DoS-limit logic via `rag.api.request_auth` and the same DI
singletons).

Two deliberate choices, documented here rather than left implicit:

- **SSE, not a WebSocket.** This is one-directional server -> client
  progress for a single already-authenticated request, exactly what SSE
  is for; a WebSocket's bidirectional channel and connection-lifecycle
  management would be unused machinery.
- **POST, not GET.** Matches `/agent/query`'s existing JSON-body request
  shape (query/filters/tenant_id/roles/as_of/require_trust_level). This
  project has no browser frontend, so trading away the browser's native
  `EventSource` API (GET-only) for a consistent request shape across both
  endpoints is the right tradeoff. Consume this endpoint via `curl -N`
  or an HTTP client's streaming mode, not `EventSource`.

`run_agent` is synchronous; it runs in Starlette's worker threadpool
(`run_in_threadpool`) so the event loop stays free to stream events as
they arrive, via an `asyncio.Queue` bridged across the thread boundary
with `call_soon_threadsafe`. A client disconnect stops this endpoint from
yielding further data, but cannot cancel the already-running agent turn
(it finishes in its worker thread regardless). A documented limitation,
not a crash risk.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from rag.agent.events import AgentEvent
from rag.agent.graph import AgentRunResult, run_agent
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
from rag.api.routers.agent_query import AgentQueryRequest, AgentQueryResponse
from rag.api.routers.query import SourceItem
from rag.config import AppConfig
from rag.embedders.base import Embedder
from rag.generation.base import LLM
from rag.retrieval.pipeline import RetrievalPipeline
from rag.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()
_limiter = get_rate_limiter()

_QUEUE_DONE = object()


def _agent_rate_limit_string() -> str:
    """Return the current `requests_per_minute` config value as a slowapi limit string.

    Same shared budget as `/agent/query`. See that router's identical
    helper for the reasoning.
    """
    return f"{get_config().security.rate_limit.requests_per_minute}/minute"


def _sse_message(event_type: str, payload: str) -> str:
    """Render one `text/event-stream` message."""
    return f"event: {event_type}\ndata: {payload}\n\n"


def _run_agent_in_thread(
    state: AgentState,
    *,
    pipeline: RetrievalPipeline,
    vectorstore: VectorStore,
    embedder: Embedder,
    llm: LLM,
    config: AppConfig,
    queue: asyncio.Queue[Any],
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Run `run_agent` synchronously, pushing events and the final result onto `queue`.

    Executed in a worker thread (via `run_in_threadpool`); `queue` belongs
    to the event loop thread, so every push is scheduled back onto the
    loop with `call_soon_threadsafe` rather than called directly.
    """

    def _on_event(event: AgentEvent) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    try:
        result = run_agent(
            state,
            pipeline=pipeline,
            vectorstore=vectorstore,
            embedder=embedder,
            llm=llm,
            config=config,
            on_event=_on_event,
        )
        loop.call_soon_threadsafe(queue.put_nowait, result)
    except Exception as exc:  # never let a worker-thread exception hang the stream
        logger.warning("Agent run failed during streaming", exc_info=True)
        loop.call_soon_threadsafe(queue.put_nowait, exc)
    finally:
        loop.call_soon_threadsafe(queue.put_nowait, _QUEUE_DONE)


def _final_response(result: AgentRunResult) -> AgentQueryResponse:
    """Build the same response shape `/agent/query` returns, for the final SSE event."""
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


async def _stream_agent_query(
    request: Request,
    state: AgentState,
    pipeline: RetrievalPipeline,
    vectorstore: VectorStore,
    embedder: Embedder,
    llm: LLM,
    config: AppConfig,
) -> AsyncIterator[str]:
    """Yield `text/event-stream` messages for one agent run, ending in `completed`/`terminated`.

    `state` is fully built (including DoS-limit validation and
    authorization-context construction) by the `_build_validated_agent_state`
    FastAPI dependency, resolved before `agent_query_stream`'s body ever
    runs -- before this async generator is even handed to
    `StreamingResponse`. An async generator's body doesn't execute at all
    until its first `__anext__()`, which happens only after Starlette has
    already sent the `http.response.start` message (status 200) for a
    `StreamingResponse`. Validating here would mean an invalid request
    gets a committed 200 status before the rejection is ever raised,
    instead of a clean 4xx.
    """
    queue: asyncio.Queue[Any] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(
        run_in_threadpool(
            _run_agent_in_thread,
            state,
            pipeline=pipeline,
            vectorstore=vectorstore,
            embedder=embedder,
            llm=llm,
            config=config,
            queue=queue,
            loop=loop,
        )
    )

    try:
        while True:
            item = await queue.get()
            if item is _QUEUE_DONE:
                break
            if await request.is_disconnected():
                break
            if isinstance(item, AgentEvent):
                if item.event_type in ("completed", "terminated"):
                    # Superseded by the richer AgentRunResult-based terminal event
                    # below (same event_type, but the full /agent/query response
                    # shape). Never emit both.
                    continue
                yield _sse_message(item.event_type, item.model_dump_json())
            elif isinstance(item, Exception):
                yield _sse_message(
                    "terminated", '{"event_type": "terminated", "termination_reason": "error"}'
                )
            elif isinstance(item, AgentRunResult):
                response = _final_response(item)
                event_type = (
                    "completed" if item.state.termination_reason == "synthesized" else "terminated"
                )
                yield _sse_message(event_type, response.model_dump_json())
    finally:
        await task


def _build_validated_agent_state(
    body: AgentQueryRequest,
    identity: VerifiedIdentity | None = Depends(get_current_identity),
    config: AppConfig = Depends(get_config),
) -> AgentState:
    """Validate the request and build its `AgentState`, before any heavy DI resolves.

    Deliberately declared as its own FastAPI dependency, listed *before*
    `pipeline`/`vectorstore`/`embedder`/`llm` in `agent_query_stream`'s
    signature, rather than as body-level checks in the route function.
    FastAPI resolves a route's declared `Depends()` parameters strictly in
    signature order and aborts the moment one of them raises, without
    resolving the parameters after it -- confirmed directly against the
    installed `fastapi==0.141.1`, not assumed (a chained-dependency
    ordering test proved a later `Depends()` callable is never invoked
    once an earlier one in the signature raises). Declaring `pipeline`/
    `vectorstore`/`embedder`/`llm` directly on the route, as this endpoint
    used to, meant FastAPI's `lru_cache`d singleton getters (see
    `api/deps.py`) were still resolved for a request this function goes
    on to reject -- cheap after the first successful request warms the
    cache, but real, unnecessary singleton construction (model load, DB
    pool open) on a cold process's very first rejected request. Putting
    the 404/422/401 checks in a dependency ahead of those four closes that
    gap without changing `/query`/`/agent/query`, which are out of this
    fix's scope.

    Parameters
    ----------
    body : AgentQueryRequest
        The request body.
    identity : VerifiedIdentity | None
        The verified caller identity, or `None` when JWT auth is disabled
        (see `get_current_identity`; also resolved ahead of the heavy
        singletons as a side effect of being this dependency's own
        sub-dependency).
    config : AppConfig
        Application configuration.

    Returns
    -------
    AgentState
        The initial agent state for `run_agent`.

    Raises
    ------
    HTTPException
        404 when `config.observability.live_events.enabled` is `False`;
        422 when `body.query`/`body.filters` violate `security.dos_limits`.
    """
    if not config.observability.live_events.enabled:
        raise HTTPException(status_code=404, detail="Live agent events are disabled")
    enforce_dos_limits(body.query, None, body.filters, config)
    auth = build_authorization_context(
        identity, body.tenant_id, body.roles, body.as_of, body.require_trust_level
    )
    return AgentState(original_query=body.query, authorization_context=auth, filters=body.filters)


@router.post("/agent/query/stream")
@_limiter.limit(_agent_rate_limit_string)
def agent_query_stream(
    request: Request,
    state: AgentState = Depends(_build_validated_agent_state),
    pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
    vectorstore: VectorStore = Depends(get_vectorstore),
    embedder: Embedder = Depends(get_embedder),
    llm: LLM = Depends(get_llm),
    config: AppConfig = Depends(get_config),
) -> StreamingResponse:
    """Stream live progress for the query in `state` through the bounded agent graph.

    Parameters
    ----------
    request : Request
        The raw HTTP request; required by `slowapi`'s rate-limit
        decorator and disconnect detection.
    state : AgentState
        The validated initial agent state, built by
        `_build_validated_agent_state` (request-body parsing,
        DoS-limit enforcement, and authorization-context construction all
        happen there, ahead of the dependencies below).
    pipeline, vectorstore, embedder, llm, config
        Same injected singletons `/agent/query` uses. Declared *after*
        `state` so a rejected request never resolves them -- see
        `_build_validated_agent_state`'s docstring.

    Returns
    -------
    StreamingResponse
        `text/event-stream`: a safe operational event per state-machine
        transition (see `rag.agent.events.AgentEvent`), ending in one
        `completed` or `terminated` event carrying the same payload shape
        as `/agent/query`'s JSON response.
    """
    return StreamingResponse(
        _stream_agent_query(request, state, pipeline, vectorstore, embedder, llm, config),
        media_type="text/event-stream",
    )
