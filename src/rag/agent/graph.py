"""Hand-rolled, explicitly-bounded agent graph driver.

Deliberately not LangGraph -- see `docs/architecture.md`'s "Agentic RAG"
section for the full rationale and the explicit, written threshold for
reconsidering that decision. The graph is a plain Python `while` loop over
node functions, each taking/returning an `AgentState`, with three
independent, plainly-inspectable integer bounds
(`max_agent_steps`/`max_retrieval_attempts`/`max_tool_calls`) rather than
one blended framework `recursion_limit`.

    START -> classify_query -> simple? --yes--> classic_rag -> final
                             --no--> decompose -> select_tool -> execute_tool
                                        -> evaluate_evidence
                                           -- sufficient --> synthesize -> final
                                           -- insufficient (bounded) --> reformulate
                                                                          -> select_tool (loop)

Every tool call's evidence passes through `RetrievalPipeline.sanitize_evidence`
before it is appended to state -- applied uniformly here, in
`_execute_tool`, regardless of which tool produced it, so no tool can
bypass field redaction/injection detection by construction.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from rag.agent import tools
from rag.agent.decisions import (
    ClassifyDecision,
    DecomposeDecision,
    EvidenceSufficiencyDecision,
    ToolSelectionDecision,
    run_decision,
)
from rag.agent.state import AgentState, Citation, ToolCallRecord
from rag.agent.tool_schemas import TOOL_ARG_MODELS
from rag.audit import log_audit_event
from rag.config import AgentConfig, AppConfig
from rag.embedders.base import Embedder
from rag.generation.base import LLM
from rag.prompts.loader import PromptTemplate, load_prompt_template
from rag.retrieval.pipeline import RetrievalPipeline, build_context
from rag.schemas import SearchResult
from rag.vectorstore.base import VectorStore

_MAX_SUBQUESTIONS = 4
_EVIDENCE_SUMMARY_CHARS = 400
_INSUFFICIENT_EVIDENCE_ANSWER = (
    "I don't have enough authorized, retrieved evidence to answer this question confidently."
)


class AgentRunResult(BaseModel):
    """The outcome of one `run_agent` call: final state plus API-facing summary fields.

    `retrieval_ms`/`generation_ms` are exact (read from `pipeline.answer()`'s
    own breakdown) on the `"classic_rag"` route. On the `"agent"` route
    they are an approximation: `retrieval_ms` sums every tool call's own
    latency, and `generation_ms` is everything else (every LLM decision
    call -- classify/decompose/select_tool/evaluate_evidence/synthesize --
    collapsed together, since they all go through the same `LLM`
    instance). `total_ms` is always exact wall-clock time.
    """

    state: AgentState
    route: Literal["classic_rag", "agent"]
    retrieval_ms: float
    generation_ms: float
    total_ms: float


@lru_cache(maxsize=16)
def _load_template(path: str) -> PromptTemplate:
    """Load and cache one agent prompt template by its resolved path."""
    return load_prompt_template(path)


def _load_templates(config: AppConfig) -> dict[str, PromptTemplate]:
    """Load all five agent decision-point templates, per `config.agent.*_prompt_path`."""
    agent_cfg = config.agent
    return {
        "classify": _load_template(
            str(config.agent_prompt_template_path(agent_cfg.classify_prompt_path))
        ),
        "decompose": _load_template(
            str(config.agent_prompt_template_path(agent_cfg.decompose_prompt_path))
        ),
        "tool_select": _load_template(
            str(config.agent_prompt_template_path(agent_cfg.tool_select_prompt_path))
        ),
        "evidence": _load_template(
            str(config.agent_prompt_template_path(agent_cfg.evidence_sufficiency_prompt_path))
        ),
        "synthesize": _load_template(
            str(config.agent_prompt_template_path(agent_cfg.synthesize_prompt_path))
        ),
    }


def _summarize_evidence(evidence: list[SearchResult]) -> str:
    """Render a short, source-labeled summary of gathered evidence for a decision prompt."""
    if not evidence:
        return "(no evidence gathered yet)"
    lines = [
        f"[{i}] {r.chunk.metadata.source}: {r.chunk.content[:_EVIDENCE_SUMMARY_CHARS]}"
        for i, r in enumerate(evidence, start=1)
    ]
    return "\n\n".join(lines)


def _accumulate_tokens(state: AgentState, llm: LLM) -> None:
    """Best-effort add the LLM's last-call token counts to the state's running total.

    Mirrors `RetrievalPipeline.answer()`'s `getattr(..., None)` pattern --
    an `LLM` implementation that doesn't track tokens contributes 0, never
    an error.
    """
    prompt = getattr(llm, "last_prompt_tokens", None)
    if prompt is not None:
        state.prompt_tokens += prompt
    completion = getattr(llm, "last_completion_tokens", None)
    if completion is not None:
        state.completion_tokens += completion


def _step_or_stop(state: AgentState, agent_cfg: AgentConfig) -> bool:
    """Increment `step_count` after a node ran; return True if the run must stop now."""
    state.step_count += 1
    if state.step_count >= agent_cfg.max_agent_steps:
        if state.termination_reason is None:
            state.termination_reason = "max_steps"
            log_audit_event("agent_max_steps_reached", steps=state.step_count)
        return True
    return False


def _classify_query(
    state: AgentState, llm: LLM, template: PromptTemplate, max_retries: int
) -> AgentState:
    """`classify_query` node: route as simple/complex. Defaults to `"simple"` on parse failure."""
    decision = run_decision(
        llm, template, ClassifyDecision, max_retries, query=state.original_query
    )
    _accumulate_tokens(state, llm)
    state.query_type = decision.query_type if decision is not None else "simple"
    return state


def _decompose(
    state: AgentState, llm: LLM, template: PromptTemplate, max_retries: int
) -> AgentState:
    """`decompose` node: split a complex query into a bounded number of subquestions."""
    decision = run_decision(
        llm, template, DecomposeDecision, max_retries, query=state.original_query
    )
    _accumulate_tokens(state, llm)
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
    here -- belt-and-suspenders on top of `SearchKnowledgeBaseArgs`'s own
    `Field(le=...)` bound, so a config change (not just the schema
    literal) is always the true final ceiling.
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
        chunks = tools.get_related_context(args, pipeline, vectorstore, state.authorization_context)
        return [SearchResult(chunk=c, score=1.0, origin="tool_fetched") for c in chunks]
    raise ValueError(f"Unknown tool: {tool_name}")  # unreachable: tool_name is Literal-validated


def _execute_tool(
    state: AgentState,
    decision: ToolSelectionDecision,
    *,
    pipeline: RetrievalPipeline,
    vectorstore: VectorStore,
    embedder: Embedder,
    dataset_id: str | None,
    agent_cfg: AgentConfig,
) -> AgentState:
    """`execute_tool` node: validate arguments, dispatch, sanitize, and record the outcome.

    Every tool's output passes through `RetrievalPipeline.sanitize_evidence`
    here -- the single, universal sanitization point (see module
    docstring) -- before being appended to `state.retrieved_evidence`. A
    validation failure or any tool-execution error is recorded as a
    failed `ToolCallRecord` and returned safely; it never propagates and
    never crashes the request.
    """
    t0 = time.perf_counter()
    arg_model = TOOL_ARG_MODELS[decision.tool_name]
    try:
        args = arg_model.model_validate(decision.tool_args)
    except ValidationError as exc:
        log_audit_event(
            "agent_tool_argument_rejected", tool_name=decision.tool_name, reason=str(exc)[:200]
        )
        state.tool_call_history.append(
            ToolCallRecord(
                tool_name=decision.tool_name,
                args={},
                result_count=0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                success=False,
                error="invalid_arguments",
            )
        )
        state.tool_call_count += 1
        return state

    try:
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
    except Exception as exc:  # tool failure must never crash the request
        state.tool_call_history.append(
            ToolCallRecord(
                tool_name=decision.tool_name,
                args=args.model_dump(),
                result_count=0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                success=False,
                error=str(exc)[:200],
            )
        )
        state.tool_call_count += 1
        return state

    sanitized = pipeline.sanitize_evidence(results, state.authorization_context)
    state.retrieved_evidence.extend(sanitized)
    if decision.tool_name == "search_knowledge_base":
        state.retrieval_attempts += 1
    state.tool_call_count += 1
    state.tool_call_history.append(
        ToolCallRecord(
            tool_name=decision.tool_name,
            args=args.model_dump(),
            result_count=len(sanitized),
            latency_ms=(time.perf_counter() - t0) * 1000,
            success=True,
        )
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
    _accumulate_tokens(state, llm)
    if decision is None:
        # Safe default on a parsing failure: proceed with whatever's gathered
        # rather than looping or crashing.
        state.evidence_sufficient = bool(state.retrieved_evidence)
        return state
    state.evidence_sufficient = decision.sufficient
    if not decision.sufficient and decision.reformulated_query:
        state.current_query = decision.reformulated_query
    return state


def _synthesize(state: AgentState, llm: LLM, template: PromptTemplate) -> AgentState:
    """`synthesize` node: render accumulated evidence into a cited final answer."""
    context = build_context(state.retrieved_evidence)
    system, user = template.render(context=context, query=state.original_query)
    state.final_answer = llm.generate(system, user)
    _accumulate_tokens(state, llm)
    state.citations = [
        Citation(
            chunk_id=r.chunk.metadata.chunk_id,
            document_id=r.chunk.metadata.document_id,
            source=r.chunk.metadata.source,
            category=r.chunk.metadata.category,
            score=r.score,
        )
        for r in state.retrieved_evidence
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
        )
        for s in result["sources"]
    ]
    state.termination_reason = "synthesized"
    return state, result


def _classic_result(state: AgentState, pipeline: RetrievalPipeline) -> AgentRunResult:
    state, result = _run_classic_rag(state, pipeline)
    return AgentRunResult(
        state=state,
        route="classic_rag",
        retrieval_ms=result["retrieval_ms"],
        generation_ms=result["generation_ms"],
        total_ms=result["total_ms"],
    )


def _agent_result(state: AgentState, t_start: float) -> AgentRunResult:
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
) -> AgentRunResult:
    """Run the bounded agent graph for one query, or the classic-RAG fast path.

    When `config.agent.enabled` is `False`, always takes the
    `"classic_rag"` route with zero extra LLM calls -- the coarse
    kill-switch matching `AuthorizationConfig`/`FieldRedactionConfig`'s
    convention. Otherwise: `classify_query` always runs first (one LLM
    decision call, even for a question that turns out to be simple -- this
    is the expected, documented cost of routing); a `"simple"` result (or
    a classify-step bound/parse failure) still falls through to the exact
    same `classic_rag` node. A `"complex"` result proceeds to `decompose`
    and the bounded `select_tool -> execute_tool -> evaluate_evidence`
    loop, gated by three independent counters
    (`max_agent_steps`/`max_retrieval_attempts`/`max_tool_calls`) — see
    the module docstring for the full graph shape.

    Parameters
    ----------
    state : AgentState
        Initial state; `original_query`/`authorization_context`/`filters`
        must already be set by the caller (see
        `rag.api.routers.agent_query`).
    pipeline, vectorstore, embedder, llm : injected singletons
        Reused, never reconstructed, from the same DI wiring
        `RetrievalPipeline`/`api.deps` already use.
    config : AppConfig
        Application configuration; `config.agent` supplies every bound.

    Returns
    -------
    AgentRunResult
        The final state plus route/timing summary fields.
    """
    t_start = time.perf_counter()
    agent_cfg = config.agent
    dataset_id = (state.filters or {}).get("dataset_id") if state.filters else None

    if not agent_cfg.enabled:
        return _classic_result(state, pipeline)

    templates = _load_templates(config)

    state = _classify_query(state, llm, templates["classify"], agent_cfg.max_json_parse_retries)
    if _step_or_stop(state, agent_cfg):
        return _agent_result(_finalize(state, llm, templates["synthesize"]), t_start)

    if state.query_type != "complex":
        return _classic_result(state, pipeline)

    state = _decompose(state, llm, templates["decompose"], agent_cfg.max_json_parse_retries)
    if _step_or_stop(state, agent_cfg):
        return _agent_result(_finalize(state, llm, templates["synthesize"]), t_start)

    while True:
        if state.tool_call_count >= agent_cfg.max_tool_calls:
            state.termination_reason = "max_tool_calls"
            log_audit_event("agent_max_tool_calls_reached", tool_calls=state.tool_call_count)
            break

        decision = _select_tool(
            state, llm, templates["tool_select"], agent_cfg.max_json_parse_retries
        )
        _accumulate_tokens(state, llm)
        if _step_or_stop(state, agent_cfg):
            break
        if decision is None:
            break

        state = _execute_tool(
            state,
            decision,
            pipeline=pipeline,
            vectorstore=vectorstore,
            embedder=embedder,
            dataset_id=dataset_id,
            agent_cfg=agent_cfg,
        )
        if _step_or_stop(state, agent_cfg):
            break

        state = _evaluate_evidence(
            state, llm, templates["evidence"], agent_cfg.max_json_parse_retries
        )
        if _step_or_stop(state, agent_cfg):
            break

        if state.evidence_sufficient:
            break
        if state.retrieval_attempts >= agent_cfg.max_retrieval_attempts:
            state.termination_reason = "max_retrieval_attempts"
            log_audit_event(
                "agent_max_retrieval_attempts_reached", attempts=state.retrieval_attempts
            )
            break
        # Otherwise current_query has been reformulated (if the decision supplied
        # one) and the loop continues back to select_tool.

    state = _finalize(state, llm, templates["synthesize"])
    return _agent_result(state, t_start)
