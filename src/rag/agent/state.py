"""Agent run state: only what the bounded graph driver needs, nothing more.

Deliberately excludes raw JWT tokens and any other unrestricted sensitive
data (see `docs/architecture.md`'s "Agentic RAG" section) -- identity is
carried only as an already-verified `AuthorizationContext`, never a
`VerifiedIdentity`/raw `sub` claim.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from rag.retrieval.authorization import AuthorizationContext
from rag.schemas import SearchResult


class Citation(BaseModel):
    """One retrieved-chunk citation surfaced in an agent run's final answer.

    Mirrors `api.routers.query.SourceItem`'s shape so the API layer can
    convert between them without reshaping data; kept as a separate model
    here so `rag.agent` never imports from `rag.api` (agent logic sits
    below the API boundary, not beside it).
    """

    chunk_id: str
    document_id: str
    source: str
    category: str | None = None
    score: float | None = None


class NodeInvocationTiming(BaseModel):
    """One graph-node call's timing, split into LLM-inference time and everything else.

    `llm_ms`/`overhead_ms` are `None` for a node that makes no direct LLM
    call (`execute_tool`) -- deliberately not `0.0`, which would misread
    as "an LLM call happened and took no time."

    Parameters
    ----------
    total_ms : float
        Total wall-clock time for this node invocation.
    llm_ms : float | None
        The portion of `total_ms` spent inside `LLM.generate()` calls
        (summed across any JSON-parse retries `run_decision` made this
        invocation) -- see `rag.agent.graph._TimingLLM`.
    overhead_ms : float | None
        `total_ms - llm_ms`: JSON parsing/validation, prompt-template
        rendering, and any other node-local work that isn't the model
        call itself.
    """

    total_ms: float
    llm_ms: float | None = None
    overhead_ms: float | None = None


class ToolCallRecord(BaseModel):
    """One tool dispatch's outcome, for observability and bounding.

    `args` holds only the validated, LLM-writable fields actually passed
    to the tool (see `rag.agent.tool_schemas`) -- never `auth`/`tenant_id`/
    `roles`, which are never LLM-writable in the first place.
    """

    tool_name: Literal[
        "search_knowledge_base", "get_document", "get_latest_document", "get_related_context"
    ]
    args: dict[str, Any] = Field(default_factory=dict)
    result_count: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: str | None = None


class AgentState(BaseModel):
    """Mutable state threaded through the hand-rolled agent graph driver.

    Parameters
    ----------
    original_query : str
        The user's original question, unmodified.
    authorization_context : AuthorizationContext | None
        The caller's verified identity/date context, set once by the API
        router before the graph runs and never reconstructed from
        anything LLM-derived. Safe to hold here -- it carries no raw
        credential (see `rag.retrieval.authorization.AuthorizationContext`).
    filters : dict[str, Any] | None
        Exact-match retrieval filters (notably `dataset_id`, the hard
        dataset namespace boundary), set once from the request and never
        mutated by any LLM decision.
    query_type : {"simple", "complex"} or None
        `classify_query`'s routing decision.
    subquestions : list[str]
        `decompose`'s output, bounded to a small number.
    current_query : str
        The query text the next tool call/decision should use -- starts
        as `original_query` or the first subquestion, and may be replaced
        by a reformulated query after an insufficient-evidence decision.
    retrieved_evidence : list[SearchResult]
        Accumulated evidence from every tool call so far, always passed
        through `RetrievalPipeline.sanitize_evidence` before being
        appended here (see `rag.agent.graph`).
    tool_call_history : list[ToolCallRecord]
        One record per tool dispatch, in order.
    retrieval_attempts : int
        Count of `search_knowledge_base` dispatches specifically -- the
        counter `max_retrieval_attempts` bounds.
    tool_call_count : int
        Count of all tool dispatches (any tool) -- the counter
        `max_tool_calls` bounds.
    step_count : int
        Count of graph node executions -- the counter `max_agent_steps`
        bounds.
    prompt_tokens, completion_tokens : int
        Best-effort accumulated token usage across every LLM call this
        run made (every decision call plus synthesis). Read via
        `getattr(llm, "last_prompt_tokens"/"last_completion_tokens",
        None)` after each call, the same pattern
        `RetrievalPipeline.answer()` uses -- `0` for an `LLM`
        implementation that doesn't track them. On a retried decision
        call (`rag.agent.decisions.run_decision`), only the last attempt's
        tokens are counted, since the LLM instance's own tracking
        attribute is overwritten per call, not accumulated -- a
        documented undercount, not a crash risk.
    evidence_sufficient : bool | None
        `evaluate_evidence`'s most recent decision.
    final_answer : str | None
        The synthesized (or insufficient-evidence) answer text.
    citations : list[Citation]
        Sources backing `final_answer`.
    termination_reason : str or None
        Why the run stopped; `None` only while still in progress.
    node_timings_ms : dict[str, list[NodeInvocationTiming]]
        Every node invocation's timing, keyed by node name (`classify`/
        `decompose`/`tool_select`/`execute_tool`/`evidence_sufficiency`/
        `synthesize`). A list, not a single value, because
        `tool_select`/`execute_tool`/`evidence_sufficiency` can run more
        than once per request (the bounded retry loop) -- both individual
        invocation timing and aggregate-across-invocations timing are
        derivable from this, never just the aggregate. Populated by
        `rag.agent.graph`'s node-timing wrapper, not by node functions
        themselves.
    node_token_usage : dict[str, dict[str, int]]
        Per-node-type `{"prompt": int, "completion": int}` token totals,
        summed across every invocation of that node type this run made.
        Same best-effort/per-call-overwrite caveat as `prompt_tokens`/
        `completion_tokens` applies per invocation.
    """

    original_query: str
    authorization_context: AuthorizationContext | None = None
    filters: dict[str, Any] | None = None
    query_type: Literal["simple", "complex"] | None = None
    subquestions: list[str] = Field(default_factory=list)
    current_query: str = ""
    retrieved_evidence: list[SearchResult] = Field(default_factory=list)
    tool_call_history: list[ToolCallRecord] = Field(default_factory=list)
    retrieval_attempts: int = 0
    tool_call_count: int = 0
    step_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    node_timings_ms: dict[str, list[NodeInvocationTiming]] = Field(default_factory=dict)
    node_token_usage: dict[str, dict[str, int]] = Field(default_factory=dict)
    evidence_sufficient: bool | None = None
    final_answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    termination_reason: (
        Literal[
            "synthesized",
            "max_steps",
            "max_retrieval_attempts",
            "max_tool_calls",
            "insufficient_evidence",
        ]
        | None
    ) = None
