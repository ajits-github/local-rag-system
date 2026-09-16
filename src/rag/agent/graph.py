"""Bounded agent orchestration for classic and agentic RAG.

The module routes simple queries to classic RAG and complex queries through
decomposition, tool use, evidence evaluation, and synthesis. Execution is
bounded by configured step, retrieval, and tool-call limits.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from functools import lru_cache, partial
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from rag.agent import mcp_client, tools
from rag.agent.decisions import (
    ClassifyDecision,
    DecomposeDecision,
    EvidenceSufficiencyDecision,
    ToolSelectionDecision,
    run_decision,
)
from rag.agent.events import AgentEvent, EventType
from rag.agent.state import AgentState, Citation, NodeInvocationTiming, ToolCallRecord
from rag.agent.tool_schemas import (
    REMOTE_MCP_TOOL_NAMES,
    TOOL_ARG_MODELS,
    WRITE_ACTION_TOOL_NAMES,
)
from rag.audit import log_audit_event
from rag.config import AgentConfig, AppConfig
from rag.embedders.base import Embedder
from rag.generation.base import LLM
from rag.observability import metrics as observability_metrics
from rag.observability import tracing
from rag.prompts.loader import PromptTemplate, load_prompt_template
from rag.retrieval.field_policy import sanitize_redaction_markers_in_answer
from rag.retrieval.pipeline import RetrievalPipeline, build_context
from rag.schemas import SearchResult
from rag.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)

_MAX_SUBQUESTIONS = 4
_EVIDENCE_SUMMARY_CHARS = 400
_INSUFFICIENT_EVIDENCE_ANSWER = (
    "I don't have enough authorized, retrieved evidence to answer this question confidently."
)

OnAgentEvent = Callable[[AgentEvent], None]


class NodeTimingStats(BaseModel):
    """Aggregate timing for one node type across every invocation a run made.

    Attributes
    ----------
    count : int
        Number of times this node ran this request.
    total_ms : float
        Sum of `total_ms` across every invocation.
    mean_ms : float
        `total_ms / count`.
    llm_ms_mean : float | None
        Mean LLM-inference-only time per invocation, or `None` for a node
        that makes no direct LLM call (`tool_execute`).
    overhead_ms_mean : float | None
        Mean non-LLM time per invocation (JSON parsing/validation/
        template rendering/tool dispatch), or `None` alongside
        `llm_ms_mean`.
    """

    count: int
    total_ms: float
    mean_ms: float
    llm_ms_mean: float | None = None
    overhead_ms_mean: float | None = None


class AgentRunResult(BaseModel):
    """Summary of one agent or classic-RAG run.

    Timing fields are exact for classic RAG. On the agent route,
    `retrieval_ms` sums tool-call latency and `generation_ms` is the
    remaining time across every LLM decision call; use `node_timings_ms`
    for a real per-node breakdown. `total_ms` is always exact.

    Attributes
    ----------
    node_timings_ms : dict[str, NodeTimingStats]
        Aggregate timing by agent node type; empty on the `"classic_rag"`
        route.
    llm_call_count : int
        Total `LLM.generate()` calls made this run.
    node_token_usage : dict[str, dict[str, int]]
        Prompt/completion token totals by node type.
    classic_sources : list[dict[str, Any]]
        Retrieved source payloads for the classic-RAG route (empty on
        `"agent"`, where the same content lives in
        `state.retrieved_evidence`).
    """

    state: AgentState
    route: Literal["classic_rag", "agent"]
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    node_timings_ms: dict[str, NodeTimingStats] = Field(default_factory=dict)
    llm_call_count: int = 0
    node_token_usage: dict[str, dict[str, int]] = Field(default_factory=dict)
    classic_sources: list[dict[str, Any]] = Field(default_factory=list)


class _TimingLLM(LLM):
    """Wrap an LLM and accumulate generation latency and call count."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm
        self.total_llm_ms: float = 0.0
        self.call_count: int = 0

    def generate(self, system: str, user: str) -> str:
        """Time one `generate()` call and forward it unchanged."""
        t0 = time.perf_counter()
        result = self._llm.generate(system, user)
        self.total_llm_ms += (time.perf_counter() - t0) * 1000
        self.call_count += 1
        return result

    def health_check(self) -> bool:
        """Forward to the wrapped instance."""
        return self._llm.health_check()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


@lru_cache(maxsize=16)
def _load_template(path: str) -> PromptTemplate:
    """Load and cache one agent prompt template by its resolved path."""
    return load_prompt_template(path)


def _load_templates(config: AppConfig) -> dict[str, PromptTemplate]:
    """Load the five agent decision-point templates from `config.agent.*_prompt_path`.

    `tool_select` resolves to one of three prompt variants depending on
    `config.mcp.client.enabled`/`config.mcp.business_actions.enabled`.
    """
    agent_cfg = config.agent
    if config.mcp.client.enabled and config.mcp.business_actions.enabled:
        tool_select_path = agent_cfg.tool_select_mcp_actions_prompt_path
    elif config.mcp.client.enabled:
        tool_select_path = agent_cfg.tool_select_mcp_prompt_path
    else:
        tool_select_path = agent_cfg.tool_select_prompt_path
    return {
        "classify": _load_template(
            str(config.agent_prompt_template_path(agent_cfg.classify_prompt_path))
        ),
        "decompose": _load_template(
            str(config.agent_prompt_template_path(agent_cfg.decompose_prompt_path))
        ),
        "tool_select": _load_template(str(config.agent_prompt_template_path(tool_select_path))),
        "evidence": _load_template(
            str(config.agent_prompt_template_path(agent_cfg.evidence_sufficiency_prompt_path))
        ),
        "synthesize": _load_template(
            str(config.agent_prompt_template_path(agent_cfg.synthesize_prompt_path))
        ),
    }


def _summarize_evidence(evidence: list[SearchResult]) -> str:
    """Render a short, source-labeled summary of gathered evidence for a decision prompt.

    Exposes `chunk_id` alongside `source` so `get_related_context` can be
    given a real chunk id to look up. No other internal metadata, such as
    document_id, dataset_id, tenant_id, or sensitive_field_ids, is exposed
    here.
    """
    if not evidence:
        return "(no evidence gathered yet)"
    lines = [
        f"[{i}] chunk_id={r.chunk.metadata.chunk_id} source={r.chunk.metadata.source}: "
        f"{r.chunk.content[:_EVIDENCE_SUMMARY_CHARS]}"
        for i, r in enumerate(evidence, start=1)
    ]
    return "\n\n".join(lines)


def _accumulate_tokens(state: AgentState, llm: LLM, node: str) -> None:
    """Best-effort add the LLM's last-call token counts to the running total.

    An `LLM` implementation that doesn't track tokens contributes 0,
    never an error. Also adds the same counts to
    `state.node_token_usage[node]` for a per-node-type breakdown.
    """
    prompt = getattr(llm, "last_prompt_tokens", None)
    completion = getattr(llm, "last_completion_tokens", None)
    if prompt is not None:
        state.prompt_tokens += prompt
    if completion is not None:
        state.completion_tokens += completion
    if prompt is not None or completion is not None:
        bucket = state.node_token_usage.setdefault(node, {"prompt": 0, "completion": 0})
        bucket["prompt"] += prompt or 0
        bucket["completion"] += completion or 0


def _call_node(
    name: str,
    state: AgentState,
    fn: Callable[[], Any],
    *,
    timed_llm: _TimingLLM | None = None,
) -> Any:
    """Record timing and tracing for one agent node invocation.

    Wraps an existing node call (`fn`) without altering its logic or
    return value.

    Parameters
    ----------
    name : str
        The node name (`classify`/`decompose`/`tool_select`/
        `tool_execute`/`evidence_sufficiency`/`synthesize`).
    state : AgentState
        The state being threaded through the graph; timing is recorded
        onto it directly.
    fn : Callable[[], Any]
        The node call to make.
    timed_llm : _TimingLLM | None, optional
        When set, the LLM-inference portion of this call is measured via
        the delta in `timed_llm.total_llm_ms` before/after `fn()` runs.
        `None` for a node that makes no direct LLM call (`tool_execute`),
        so `llm_ms`/`overhead_ms` are recorded as `None` rather than a
        misleading `0.0`.

    Returns
    -------
    Any
        Whatever `fn()` returned, unchanged.
    """
    llm_ms_before = timed_llm.total_llm_ms if timed_llm is not None else None
    t0 = time.perf_counter()
    with tracing.start_span(name) as span:
        result = fn()
        total_ms = (time.perf_counter() - t0) * 1000
        llm_ms = (
            (timed_llm.total_llm_ms - llm_ms_before)
            if timed_llm is not None and llm_ms_before is not None
            else None
        )
        overhead_ms = max(total_ms - llm_ms, 0.0) if llm_ms is not None else None
        tracing.set_attributes(
            span, {"node": name, "total_ms": total_ms, "llm_ms": llm_ms, "overhead_ms": overhead_ms}
        )
    state.node_timings_ms.setdefault(name, []).append(
        NodeInvocationTiming(total_ms=total_ms, llm_ms=llm_ms, overhead_ms=overhead_ms)
    )
    observability_metrics.observe_node_latency(
        name, total_ms / 1000, llm_ms / 1000 if llm_ms is not None else None
    )
    return result


def _aggregate_node_timings(
    node_timings_ms: dict[str, list[NodeInvocationTiming]],
) -> dict[str, NodeTimingStats]:
    """Reduce per-invocation node timings into one `NodeTimingStats` per node type."""
    stats: dict[str, NodeTimingStats] = {}
    for name, invocations in node_timings_ms.items():
        count = len(invocations)
        total_ms = sum(t.total_ms for t in invocations)
        llm_values = [t.llm_ms for t in invocations if t.llm_ms is not None]
        overhead_values = [t.overhead_ms for t in invocations if t.overhead_ms is not None]
        stats[name] = NodeTimingStats(
            count=count,
            total_ms=total_ms,
            mean_ms=total_ms / count if count else 0.0,
            llm_ms_mean=(sum(llm_values) / len(llm_values)) if llm_values else None,
            overhead_ms_mean=(
                (sum(overhead_values) / len(overhead_values)) if overhead_values else None
            ),
        )
    return stats


def _log_tool_call_completed(
    tool_name: str,
    *,
    success: bool,
    result_count: int,
    latency_ms: float,
    is_remote: bool,
    error_type: str | None = None,
) -> None:
    """Emit one structured tool-completion log line.

    Never includes raw tool arguments, raw tool output, or the exception
    message text -- only the tool name, a bounded `error_type` (the
    exception class name, never `str(exc)`), and counts/timing.
    """
    logger.info(
        "tool_call_completed",
        extra={
            "tool_name": tool_name,
            "success": success,
            "result_count": result_count,
            "duration_ms": round(latency_ms, 2),
            "origin": "remote" if is_remote else "local",
            "error_type": error_type,
        },
    )


def _emit_event(
    on_event: OnAgentEvent | None,
    event_type: EventType,
    state: AgentState,
    *,
    t_start: float | None = None,
    tool_name: str | None = None,
    result_count: int | None = None,
    route: str | None = None,
) -> None:
    """Emit one safe live-progress event, if a sink is set.

    Never raises. A broken consumer callback can never crash the run.
    Carries only bounded, already-safe metadata; never chain-of-thought,
    raw prompts, retrieved content, or credentials.
    """
    if on_event is None:
        return
    try:
        on_event(
            AgentEvent(
                event_type=event_type,
                step=state.step_count,
                tool_name=tool_name,
                elapsed_ms=(time.perf_counter() - t_start) * 1000 if t_start is not None else None,
                retrieved_chunk_count=result_count,
                evidence_sufficient=state.evidence_sufficient,
                termination_reason=state.termination_reason,
                route=route,
            )
        )
    except Exception:
        logger.warning("on_event callback raised; continuing", exc_info=True)


def _increment_step(state: AgentState) -> None:
    """Count one graph-node execution toward `max_agent_steps`."""
    state.step_count += 1


def _check_step_bound(state: AgentState, agent_cfg: AgentConfig) -> bool:
    """Return True and label `max_steps` (once) once the step ceiling is reached.

    Does not increment `step_count`; callers must call `_increment_step`
    first. Split out so a node reaching a real terminal condition on the
    last allowed step isn't mislabeled `max_steps`.
    """
    if state.step_count >= agent_cfg.max_agent_steps:
        if state.termination_reason is None:
            state.termination_reason = "max_steps"
            log_audit_event("agent_max_steps_reached", steps=state.step_count)
        return True
    return False


def _step_or_stop(state: AgentState, agent_cfg: AgentConfig) -> bool:
    """Increment `step_count` after a node ran; return True if the run must stop now."""
    _increment_step(state)
    return _check_step_bound(state, agent_cfg)


_CASE_MUTATION_VERB_RE = re.compile(
    r"\b(close|closes|closing|reopen|reopens|reopening|resolve|resolves|resolving)\b"
)
_CASE_MUTATION_DIRECTIVE_RE = re.compile(
    r"\b(set|sets|setting|change|changes|changing|update|updates|updating|"
    r"mark|marks|marking|move|moves|moving|transition|transitions|transitioning)\b"
)
_CASE_STATUS_TARGET_RE = re.compile(r"\b(status|state|open|in.progress|resolved|closed)\b")


def _looks_like_case_mutation_request(query: str) -> bool:
    """Return whether the query clearly requests a case-status mutation.

    Matching requests are routed through the agent path so server-side
    authorization, transition, and approval rules can run; this only
    ever widens routing toward the agent path, never infers authorization
    or approval itself.
    """
    lowered = query.lower()
    if "case" not in lowered:
        return False
    if _CASE_MUTATION_VERB_RE.search(lowered):
        return True
    return bool(
        _CASE_MUTATION_DIRECTIVE_RE.search(lowered) and _CASE_STATUS_TARGET_RE.search(lowered)
    )


def _classify_query(
    state: AgentState, llm: LLM, template: PromptTemplate, max_retries: int
) -> AgentState:
    """Classify the query and force explicit case mutations onto the agent route."""
    decision = run_decision(
        llm, template, ClassifyDecision, max_retries, query=state.original_query
    )
    _accumulate_tokens(state, llm, "classify")
    query_type = decision.query_type if decision is not None else "simple"
    if query_type != "complex" and _looks_like_case_mutation_request(state.original_query):
        query_type = "complex"
    state.query_type = query_type
    return state


def _decompose(
    state: AgentState, llm: LLM, template: PromptTemplate, max_retries: int
) -> AgentState:
    """`decompose` node: split a complex query into a bounded number of subquestions."""
    decision = run_decision(
        llm, template, DecomposeDecision, max_retries, query=state.original_query
    )
    _accumulate_tokens(state, llm, "decompose")
    subquestions = (decision.subquestions if decision is not None else [])[:_MAX_SUBQUESTIONS]
    state.subquestions = subquestions
    state.current_query = subquestions[0] if subquestions else state.original_query
    return state


def _select_tool(
    state: AgentState, llm: LLM, template: PromptTemplate, max_retries: int
) -> ToolSelectionDecision | None:
    """`select_tool` node: decide which tool to dispatch next, and with what raw arguments."""
    return run_decision(
        llm,
        template,
        ToolSelectionDecision,
        max_retries,
        query=state.current_query,
        evidence_summary=_summarize_evidence(state.retrieved_evidence),
    )


def _dispatch_tool(
    tool_name: str,
    args: Any,
    *,
    state: AgentState,
    pipeline: RetrievalPipeline,
    vectorstore: VectorStore,
    embedder: Embedder,
    dataset_id: str | None,
    agent_cfg: AgentConfig,
) -> list[SearchResult]:
    """Call the matching `rag.agent.tools` function and wrap its output as `SearchResult`s.

    `search_knowledge_base`'s `top_k` is clamped to `agent_cfg.max_tool_top_k`
    here, in addition to `SearchKnowledgeBaseArgs`'s own `Field(le=...)`
    bound, so the runtime config is always the final ceiling.
    """
    if tool_name == "search_knowledge_base":
        clamped_top_k = min(args.top_k, agent_cfg.max_tool_top_k)
        clamped_args = args.model_copy(update={"top_k": clamped_top_k})
        return list(
            tools.search_knowledge_base(
                clamped_args, pipeline, state.filters, state.authorization_context
            )
        )
    if tool_name == "get_document":
        chunks = tools.get_document(
            args,
            pipeline,
            vectorstore,
            dataset_id,
            state.current_query,
            embedder,
            state.authorization_context,
            agent_cfg.max_chunks_per_document_fetch,
            agent_cfg.max_chunks_per_document_fetch_hard_ceiling,
        )
        return [SearchResult(chunk=c, score=1.0, origin="tool_fetched") for c in chunks]
    if tool_name == "get_latest_document":
        chunks = tools.get_latest_document(
            args,
            pipeline,
            vectorstore,
            dataset_id,
            state.current_query,
            embedder,
            state.authorization_context,
            agent_cfg.max_chunks_per_document_fetch,
            agent_cfg.max_chunks_per_document_fetch_hard_ceiling,
        )
        return [SearchResult(chunk=c, score=1.0, origin="tool_fetched") for c in chunks]
    if tool_name == "get_related_context":
        chunks = tools.get_related_context(
            args, pipeline, vectorstore, state.authorization_context, dataset_id
        )
        return [SearchResult(chunk=c, score=1.0, origin="tool_fetched") for c in chunks]
    raise ValueError(f"Unknown tool: {tool_name}")  # unreachable: tool_name is Literal-validated


def _dispatch_mcp_tool(
    tool_name: str,
    args: Any,
    *,
    state: AgentState,
    config: AppConfig,
    mcp_app: Any | None,
) -> list[SearchResult]:
    """Dispatch one remote MCP business tool via `rag.agent.mcp_client`."""
    return mcp_client.dispatch_remote_tool_sync(
        tool_name,
        args,
        auth=state.authorization_context,
        config=config,
        mcp_app=mcp_app,
        case_approvals=state.case_approvals,
    )


def _execute_tool(
    state: AgentState,
    decision: ToolSelectionDecision,
    *,
    pipeline: RetrievalPipeline,
    vectorstore: VectorStore,
    embedder: Embedder,
    dataset_id: str | None,
    agent_cfg: AgentConfig,
    config: AppConfig,
    mcp_app: Any | None = None,
    on_event: OnAgentEvent | None = None,
) -> AgentState:
    """Validate, dispatch, sanitize, and record one tool call.

    Tool output is sanitized via `RetrievalPipeline.sanitize_evidence`
    using the resolved authorization context before entering agent
    evidence, so no tool can bypass redaction. Validation and execution
    failures are recorded as a failed tool call rather than propagated.
    """
    t0 = time.perf_counter()
    arg_model = TOOL_ARG_MODELS[decision.tool_name]
    try:
        args = arg_model.model_validate(decision.tool_args)
    except ValidationError as exc:
        # Pydantic v2's default str(ValidationError) embeds each offending
        # field's (truncated) input_value -- tool args originate from the
        # LLM's own JSON decision, itself potentially influenced by
        # prompt-injected retrieved content, so that could leak a bounded
        # fragment of adversarial/retrieved text into the audit log.
        # errors(include_input=False) reports only the error shape
        # (field path + error type), matching the general tool-dispatch
        # failure path below, which logs only error_type for the same reason.
        error_shape = [
            {"loc": e["loc"], "type": e["type"]} for e in exc.errors(include_input=False)
        ]
        log_audit_event(
            "agent_tool_argument_rejected",
            tool_name=decision.tool_name,
            reason=str(error_shape)[:200],
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        state.tool_call_history.append(
            ToolCallRecord(
                tool_name=decision.tool_name,
                args={},
                result_count=0,
                latency_ms=latency_ms,
                success=False,
                error="invalid_arguments",
            )
        )
        state.tool_call_count += 1
        observability_metrics.observe_tool_call(decision.tool_name, False, latency_ms / 1000)
        observability_metrics.observe_error("tool")
        _log_tool_call_completed(
            decision.tool_name,
            success=False,
            result_count=0,
            latency_ms=latency_ms,
            is_remote=decision.tool_name in REMOTE_MCP_TOOL_NAMES,
            error_type="invalid_arguments",
        )
        return state

    is_remote_tool = decision.tool_name in REMOTE_MCP_TOOL_NAMES
    is_write_action = decision.tool_name in WRITE_ACTION_TOOL_NAMES
    mcp_client_disabled = is_remote_tool and not config.mcp.client.enabled
    business_action_disabled = is_write_action and not config.mcp.business_actions.enabled
    if mcp_client_disabled or business_action_disabled:
        # Defense in depth: the tool-select prompt already never offers a name
        # unless its config is enabled (see _load_templates), so this only fires
        # on a hallucinated/stale decision. Fails like an invalid-argument decision:
        # a recorded failure, never a dispatch attempt, never a crash.
        error = "mcp_client_disabled" if mcp_client_disabled else "business_actions_disabled"
        log_audit_event(
            "agent_mcp_tool_disabled" if mcp_client_disabled else "agent_business_action_disabled",
            tool_name=decision.tool_name,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        state.tool_call_history.append(
            ToolCallRecord(
                tool_name=decision.tool_name,
                args=args.model_dump(),
                result_count=0,
                latency_ms=latency_ms,
                success=False,
                error=error,
            )
        )
        state.tool_call_count += 1
        observability_metrics.observe_tool_call(decision.tool_name, False, latency_ms / 1000)
        _log_tool_call_completed(
            decision.tool_name,
            success=False,
            result_count=0,
            latency_ms=latency_ms,
            is_remote=is_remote_tool,
            error_type=error,
        )
        return state

    _emit_event(on_event, "tool_started", state, tool_name=decision.tool_name)
    try:
        with tracing.start_span(
            decision.tool_name, attributes={"tool_name": decision.tool_name}
        ) as span:
            if is_remote_tool:
                results = _dispatch_mcp_tool(
                    decision.tool_name, args, state=state, config=config, mcp_app=mcp_app
                )
            else:
                results = _dispatch_tool(
                    decision.tool_name,
                    args,
                    state=state,
                    pipeline=pipeline,
                    vectorstore=vectorstore,
                    embedder=embedder,
                    dataset_id=dataset_id,
                    agent_cfg=agent_cfg,
                )
            tracing.set_attributes(span, {"tool_success": True, "result_count": len(results)})
    except Exception as exc:  # tool failure must never crash the request
        latency_ms = (time.perf_counter() - t0) * 1000
        state.tool_call_history.append(
            ToolCallRecord(
                tool_name=decision.tool_name,
                args=args.model_dump(),
                result_count=0,
                latency_ms=latency_ms,
                success=False,
                error=str(exc)[:200],
            )
        )
        state.tool_call_count += 1
        observability_metrics.observe_tool_call(decision.tool_name, False, latency_ms / 1000)
        observability_metrics.observe_error("tool")
        _log_tool_call_completed(
            decision.tool_name,
            success=False,
            result_count=0,
            latency_ms=latency_ms,
            is_remote=is_remote_tool,
            error_type=type(exc).__name__,
        )
        _emit_event(on_event, "tool_completed", state, tool_name=decision.tool_name, result_count=0)
        return state

    effective_auth = pipeline.resolve_auth(
        state.authorization_context, {"dataset_id": dataset_id} if dataset_id else None
    )
    sanitized = pipeline.sanitize_evidence(results, effective_auth)
    state.retrieved_evidence.extend(sanitized)
    if decision.tool_name == "search_knowledge_base":
        state.retrieval_attempts += 1
    state.tool_call_count += 1
    latency_ms = (time.perf_counter() - t0) * 1000
    state.tool_call_history.append(
        ToolCallRecord(
            tool_name=decision.tool_name,
            args=args.model_dump(),
            result_count=len(sanitized),
            latency_ms=latency_ms,
            success=True,
        )
    )
    observability_metrics.observe_tool_call(decision.tool_name, True, latency_ms / 1000)
    _log_tool_call_completed(
        decision.tool_name,
        success=True,
        result_count=len(sanitized),
        latency_ms=latency_ms,
        is_remote=is_remote_tool,
    )
    _emit_event(
        on_event, "tool_completed", state, tool_name=decision.tool_name, result_count=len(sanitized)
    )
    return state


def _evaluate_evidence(
    state: AgentState, llm: LLM, template: PromptTemplate, max_retries: int
) -> AgentState:
    """`evaluate_evidence` node: decide if gathered evidence suffices, or reformulate."""
    decision = run_decision(
        llm,
        template,
        EvidenceSufficiencyDecision,
        max_retries,
        query=state.original_query,
        evidence_summary=_summarize_evidence(state.retrieved_evidence),
    )
    _accumulate_tokens(state, llm, "evidence_sufficiency")
    if decision is None:
        # Safe default on a parsing failure: proceed with whatever's gathered
        # rather than looping or crashing.
        state.evidence_sufficient = bool(state.retrieved_evidence)
        return state
    state.evidence_sufficient = decision.sufficient
    if not decision.sufficient and decision.reformulated_query:
        state.current_query = decision.reformulated_query
    return state


def _order_evidence_for_synthesis(evidence: list[SearchResult]) -> list[SearchResult]:
    """Stable-sort evidence so authoritative/untagged sources precede untrusted ones.

    Synthesis-only reordering: `state.retrieved_evidence` itself (read by
    every earlier decision prompt) is left untouched. Without this, an
    untrusted source gathered first could outrank an authoritative source
    on the same fact purely due to retrieval order. The sort is stable
    and never drops or adds a source, only its `[Source N]` numbering.
    """
    return sorted(
        evidence, key=lambda r: (r.chunk.metadata.trust_level or "").lower() == "untrusted"
    )


def _synthesize(state: AgentState, llm: LLM, template: PromptTemplate) -> AgentState:
    """Render accumulated evidence into a cited final answer.

    Evidence is reordered first (see `_order_evidence_for_synthesis`), and
    the generated answer is passed through
    `sanitize_redaction_markers_in_answer` before being stored.
    """
    ordered_evidence = _order_evidence_for_synthesis(state.retrieved_evidence)
    context = build_context(ordered_evidence)
    system, user = template.render(context=context, query=state.original_query)
    state.final_answer = sanitize_redaction_markers_in_answer(llm.generate(system, user))
    _accumulate_tokens(state, llm, "synthesize")
    state.citations = [
        Citation(
            chunk_id=r.chunk.metadata.chunk_id,
            document_id=r.chunk.metadata.document_id,
            source=r.chunk.metadata.source,
            category=r.chunk.metadata.category,
            score=r.score,
            content_type=r.chunk.metadata.content_type,
            section_path=r.chunk.metadata.section_path,
            page=r.chunk.metadata.page,
            attachment_name=r.chunk.metadata.attachment_name,
            source_anchor=r.chunk.metadata.source_anchor,
            vision_generated=r.chunk.metadata.vision_generated,
            origin=r.origin,
        )
        for r in ordered_evidence
    ]
    if state.termination_reason is None:
        state.termination_reason = "synthesized"
    return state


def _insufficient_evidence_response(state: AgentState) -> AgentState:
    """Terminal node when no usable evidence was ever gathered: no further LLM call."""
    state.final_answer = _INSUFFICIENT_EVIDENCE_ANSWER
    state.citations = []
    if state.termination_reason is None:
        state.termination_reason = "insufficient_evidence"
    return state


def _finalize(state: AgentState, llm: LLM, synthesize_template: PromptTemplate) -> AgentState:
    """Route to `synthesize` if evidence was gathered, else the insufficient-evidence response."""
    if state.retrieved_evidence:
        return _synthesize(state, llm, synthesize_template)
    return _insufficient_evidence_response(state)


def _finalize_timed(
    state: AgentState,
    timed_llm: _TimingLLM,
    templates: dict[str, PromptTemplate],
    on_event: OnAgentEvent | None,
    t_start: float,
) -> AgentState:
    """`run_agent`'s entrypoint into `_finalize`, timing the `synthesize` node when it runs."""
    if state.retrieved_evidence:
        _emit_event(on_event, "synthesis_started", state, t_start=t_start)
        return _call_node(
            "synthesize",
            state,
            lambda: _synthesize(state, timed_llm, templates["synthesize"]),
            timed_llm=timed_llm,
        )
    return _insufficient_evidence_response(state)


def _run_classic_rag(
    state: AgentState, pipeline: RetrievalPipeline
) -> tuple[AgentState, dict[str, Any]]:
    """`classic_rag` node: the unmodified `RetrievalPipeline.answer()` fast path."""
    result = pipeline.answer(
        state.original_query, filters=state.filters, auth=state.authorization_context
    )
    state.query_type = state.query_type or "simple"
    state.prompt_tokens += result.get("prompt_tokens") or 0
    state.completion_tokens += result.get("completion_tokens") or 0
    state.final_answer = result["answer"]
    state.citations = [
        Citation(
            chunk_id=s["chunk_id"],
            document_id=s["document_id"],
            source=s["source"],
            category=s.get("category"),
            score=s.get("score"),
            content_type=s.get("content_type"),
            section_path=s.get("section_path"),
            page=s.get("page"),
            attachment_name=s.get("attachment_name"),
            source_anchor=s.get("source_anchor"),
            vision_generated=bool(s.get("vision_generated", False)),
            origin=s.get("origin", "retrieved"),
        )
        for s in result["sources"]
    ]
    state.termination_reason = "synthesized"
    return state, result


def _classic_result(state: AgentState, pipeline: RetrievalPipeline) -> AgentRunResult:
    """Run classic RAG and wrap the result as an `AgentRunResult`."""
    state, result = _run_classic_rag(state, pipeline)
    return AgentRunResult(
        state=state,
        route="classic_rag",
        retrieval_ms=result["retrieval_ms"],
        generation_ms=result["generation_ms"],
        total_ms=result["total_ms"],
        classic_sources=result["sources"],
    )


def _agent_result(state: AgentState, t_start: float) -> AgentRunResult:
    """Build an `AgentRunResult` for a completed `"agent"`-route run."""
    total_ms = (time.perf_counter() - t_start) * 1000
    retrieval_ms = sum(record.latency_ms for record in state.tool_call_history)
    generation_ms = max(total_ms - retrieval_ms, 0.0)
    return AgentRunResult(
        state=state,
        route="agent",
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        total_ms=total_ms,
    )


def run_agent(
    state: AgentState,
    *,
    pipeline: RetrievalPipeline,
    vectorstore: VectorStore,
    embedder: Embedder,
    llm: LLM,
    config: AppConfig,
    mcp_app: Any | None = None,
    on_event: OnAgentEvent | None = None,
    cancel_event: threading.Event | None = None,
) -> AgentRunResult:
    """Run one bounded RAG request.

    Simple queries use the classic-RAG path. Complex queries use
    decomposition, tool selection, bounded execution, evidence
    evaluation, and synthesis. `config.agent.enabled=False` always takes
    the classic-RAG path with no extra LLM calls.

    Parameters
    ----------
    state : AgentState
        Initial state; `original_query`/`authorization_context`/`filters`
        must already be set.
    pipeline, vectorstore, embedder, llm : injected singletons
        Reused from the existing DI wiring.
    config : AppConfig
        `config.agent` supplies every bound.
    mcp_app : Any | None, optional
        The in-process MCP ASGI app, threaded through for the remote
        business tools; unused when `mcp.client.enabled=False`.
    on_event : OnAgentEvent | None, optional
        Called with a safe `AgentEvent` at each state transition, if set.
        Never raises out of `run_agent` even if the callback itself does.
    cancel_event : threading.Event | None, optional
        A cooperative cancellation signal, checked once per iteration of
        the bounded tool-call loop (between `tool_select`/`tool_execute`/
        `evidence_sufficiency` node executions). When set, the run stops
        as soon as the current iteration's checkpoint is reached, labels
        `termination_reason="cancelled"`, and skips the final synthesis
        LLM call entirely (unlike every other termination reason, which
        still makes a best-effort synthesis call). Intended for a caller
        (e.g. `POST /agent/query/stream`) that detects its client
        disconnected and no longer needs the answer. `None` (the default)
        never cancels. Does not interrupt a node call already in
        progress; cancellation is checked only between iterations.

    Returns
    -------
    AgentRunResult
        The final state plus route/timing summary fields.
    """
    t_start = time.perf_counter()
    agent_cfg = config.agent
    dataset_id = (state.filters or {}).get("dataset_id") if state.filters else None
    timed_llm = _TimingLLM(llm)

    root_span_cm = tracing.start_span("agent_query")
    root_span = root_span_cm.__enter__()

    def _finish(result: AgentRunResult) -> AgentRunResult:
        """Close the root span, record run-level metrics/events, and attach timing summaries."""
        tracing.set_attributes(
            root_span,
            {
                "route": result.route,
                "termination_reason": result.state.termination_reason,
                "agent_step_count": result.state.step_count,
                "tool_call_count": result.state.tool_call_count,
            },
        )
        # Logged while root_span is still active (before __exit__ below detaches
        # it) so JSONFormatter's ambient trace_id/span_id lookup finds it. Never
        # includes query/answer/evidence text -- only run-shape counters.
        logger.info(
            "agent_request_completed",
            extra={
                "route": result.route,
                "termination_reason": result.state.termination_reason,
                "step_count": result.state.step_count,
                "tool_call_count": result.state.tool_call_count,
                "retrieval_attempts": result.state.retrieval_attempts,
                "evidence_sufficient": result.state.evidence_sufficient,
                "duration_ms": round(result.total_ms, 2),
                "success": result.state.termination_reason == "synthesized",
            },
        )
        try:
            root_span_cm.__exit__(None, None, None)
        except Exception:
            logger.warning("Failed to close agent_query span", exc_info=True)

        observability_metrics.observe_agent_request(
            result.route, result.total_ms / 1000, result.state.step_count
        )
        if result.state.termination_reason is not None:
            observability_metrics.observe_termination_reason(result.state.termination_reason)
        if result.state.evidence_sufficient is not None:
            observability_metrics.observe_evidence_sufficiency(result.state.evidence_sufficient)

        _emit_event(
            on_event,
            "completed" if result.state.termination_reason == "synthesized" else "terminated",
            result.state,
            t_start=t_start,
            route=result.route,
        )
        return result.model_copy(
            update={
                "node_timings_ms": _aggregate_node_timings(result.state.node_timings_ms),
                "node_token_usage": result.state.node_token_usage,
                "llm_call_count": timed_llm.call_count if result.route == "agent" else 1,
            }
        )

    _emit_event(on_event, "query_received", state, t_start=t_start)

    if not agent_cfg.enabled:
        _emit_event(on_event, "route_selected", state, t_start=t_start, route="classic_rag")
        return _finish(_classic_result(state, pipeline))

    templates = _load_templates(config)

    state = _call_node(
        "classify",
        state,
        lambda: _classify_query(
            state, timed_llm, templates["classify"], agent_cfg.max_json_parse_retries
        ),
        timed_llm=timed_llm,
    )
    if _step_or_stop(state, agent_cfg):
        state = _finalize_timed(state, timed_llm, templates, on_event, t_start)
        return _finish(_agent_result(state, t_start))

    if state.query_type != "complex":
        _emit_event(on_event, "route_selected", state, t_start=t_start, route="classic_rag")
        return _finish(_classic_result(state, pipeline))

    _emit_event(on_event, "route_selected", state, t_start=t_start, route="agent")
    _emit_event(on_event, "decomposition_started", state, t_start=t_start)
    state = _call_node(
        "decompose",
        state,
        lambda: _decompose(
            state, timed_llm, templates["decompose"], agent_cfg.max_json_parse_retries
        ),
        timed_llm=timed_llm,
    )
    _emit_event(on_event, "decomposition_completed", state, t_start=t_start)
    if _step_or_stop(state, agent_cfg):
        state = _finalize_timed(state, timed_llm, templates, on_event, t_start)
        return _finish(_agent_result(state, t_start))

    while True:
        if cancel_event is not None and cancel_event.is_set():
            # Cooperative cancellation checkpoint: the caller (e.g. a disconnected
            # SSE client) no longer needs this run's answer. Never interrupts a
            # node call already in progress, only stops the next one from
            # starting -- so this bounds, rather than eliminates, wasted work.
            state.termination_reason = "cancelled"
            break

        if state.tool_call_count >= agent_cfg.max_tool_calls:
            state.termination_reason = "max_tool_calls"
            log_audit_event("agent_max_tool_calls_reached", tool_calls=state.tool_call_count)
            break

        decision = _call_node(
            "tool_select",
            state,
            partial(
                _select_tool,
                state,
                timed_llm,
                templates["tool_select"],
                agent_cfg.max_json_parse_retries,
            ),
            timed_llm=timed_llm,
        )
        _accumulate_tokens(state, timed_llm, "tool_select")
        if _step_or_stop(state, agent_cfg):
            break
        if decision is None:
            break
        _emit_event(on_event, "tool_selected", state, t_start=t_start, tool_name=decision.tool_name)

        state = _call_node(
            "tool_execute",
            state,
            partial(
                _execute_tool,
                state,
                decision,
                pipeline=pipeline,
                vectorstore=vectorstore,
                embedder=embedder,
                dataset_id=dataset_id,
                agent_cfg=agent_cfg,
                config=config,
                mcp_app=mcp_app,
                on_event=on_event,
            ),
        )
        if _step_or_stop(state, agent_cfg):
            break
        if decision.tool_name in WRITE_ACTION_TOOL_NAMES:
            # A write-action attempt (any outcome: executed, approval_required,
            # invalid_transition, already_in_status, or a dispatch failure) is
            # always the last tool call of a run -- never retried in the same
            # run's evidence-sufficiency/reformulate loop. Proceeds straight to
            # synthesis, which reports the true outcome from the evidence
            # _execute_tool already recorded.
            break

        state = _call_node(
            "evidence_sufficiency",
            state,
            partial(
                _evaluate_evidence,
                state,
                timed_llm,
                templates["evidence"],
                agent_cfg.max_json_parse_retries,
            ),
            timed_llm=timed_llm,
        )
        _emit_event(on_event, "evidence_evaluated", state, t_start=t_start)
        _increment_step(state)

        if state.evidence_sufficient:
            # Evidence became sufficient on the final allowed step: a real
            # convergence, not a step-bound cutoff, so check it before
            # _check_step_bound would otherwise mislabel this run "max_steps".
            break
        if _check_step_bound(state, agent_cfg):
            break
        if state.retrieval_attempts >= agent_cfg.max_retrieval_attempts:
            state.termination_reason = "max_retrieval_attempts"
            log_audit_event(
                "agent_max_retrieval_attempts_reached", attempts=state.retrieval_attempts
            )
            break
        _emit_event(on_event, "retry_started", state, t_start=t_start)
        # Otherwise current_query has been reformulated (if the decision supplied
        # one) and the loop continues back to select_tool.

    if state.termination_reason == "cancelled":
        # Skip the final synthesis LLM call entirely: nobody is waiting on the
        # answer, so making one more model call would defeat the point of
        # cancelling. Every other termination reason still makes a best-effort
        # synthesis call from whatever evidence was gathered.
        state.final_answer = None
        state.citations = []
    else:
        state = _finalize_timed(state, timed_llm, templates, on_event, t_start)
    return _finish(_agent_result(state, t_start))
