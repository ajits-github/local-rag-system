"""Typed shared graph state for the multi-agent experiment.

Field-ownership discipline is the load-bearing design decision here: every
field is written by exactly one specialist branch (`knowledge_*` only by
`knowledge_agent`, `case_*`/`pending_action`/`case_action_outcome` only by
the reused business-branch nodes), so two parallel branches never race to
write the same key -- no custom merge/conflict-resolution logic is needed
for the vast majority of this state. The three fields that genuinely can
be written more than once across a run's lifetime (`knowledge_tool_calls`,
`business_tool_calls`, and `citations`... see below) are the exception,
and are handled explicitly:

- `knowledge_tool_calls`/`business_tool_calls` use LangGraph's
  `operator.add` reducer, since the bounded knowledge-retry loop
  (`evidence_critic` -> `coordinator` -> `knowledge_agent`, see
  `nodes.py`) invokes `knowledge_agent` more than once per run, and each
  invocation's tool-call record should accumulate, not overwrite the
  prior attempt's record.
- `citations`/`tool_call_log` are deliberately *not* reducers: `merge`
  recomputes both from scratch on every invocation (reading the current,
  possibly-just-retried `knowledge_evidence`), so plain last-write-wins
  is correct there -- accumulating would duplicate citations across a
  retry.

This mirrors `langgraph_experiment.state.GraphState`'s own "plain,
JSON-safe values, no shared mutable object" philosophy; see that
module's docstring for the broader rationale (human-readable checkpoints,
no coupling to `rag`'s pydantic model versions). Several field names
(`original_query`, `caller_subject`, `tenant_id`, `roles`, `dataset_id`,
`case_id`, `case_tool_name`, `case_result`, `case_found`,
`requested_new_status`, `pending_action`, `approval_decision`,
`case_action_outcome`, `termination_reason`, `citations`) are kept
*identical* to `langgraph_experiment.state.GraphState`'s own field names
on purpose: the business branch's node functions (`select_case_tool`,
`make_execute_case_read_node`, `validate_write_request`,
`make_wait_for_approval_node`, `make_execute_write_action_node`,
`synthesize_write_result`) are imported and reused **unmodified** from
`langgraph_experiment.nodes` (see `nodes.py`'s module docstring) -- they
are plain functions of `(state) -> dict` that read/write these keys by
name, so matching the key names is what makes that reuse possible without
touching a single line of already-tested code.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langgraph_experiment.state import Citation, EvidenceItem, PendingCaseAction

Route = Literal["knowledge_only", "business_read_only", "mixed", "business_write"]
SpecialistStatus = Literal["not_run", "ok", "no_evidence", "not_found_or_denied", "failed"]


class ToolCallRecord(TypedDict):
    """One specialist's tool dispatch, for observability and the evaluation harness.

    Deliberately shallow: no query text, no chunk content, no case detail
    -- only what `docs`/eval metrics need (which specialist, which tool,
    did it succeed, how long it took). Mirrors the restraint
    `rag.agent.state.ToolCallRecord`/the `tool_call_completed` log line
    already apply in production (see CLAUDE.md's "Logging" section).
    """

    specialist: Literal["knowledge", "business"]
    tool_name: str
    success: bool
    latency_ms: float


class MultiAgentState(TypedDict, total=False):
    """The multi-agent graph's full state schema.

    Attributes
    ----------
    original_query : str
        The caller's question, unmodified. Required at graph start.
    thread_id, caller_subject, tenant_id, roles, dataset_id, workflow_started_at
        Demo caller identity/scope plus the production-hardening fields
        the reused business-write branch now requires, identical in shape
        and purpose to `langgraph_experiment.state.GraphState`'s
        same-named fields -- `thread_id` is a display/audit-only copy of
        the LangGraph thread id (never used for routing/authorization),
        and `workflow_started_at` is stamped once, idempotently, by
        `orchestrator.coordinator`'s round-0 branch (see that node's
        docstring) since `business_agent`'s reused `wait_for_approval`
        node reads it unconditionally to enforce `GraphDeps.
        max_workflow_duration_seconds`.
    coordinator_round : int
        How many times `coordinator` has run this thread (starts at 0
        before its first run; `coordinator` itself sets it to 1, then to
        `round + 1` on a critic-requested retry). Bounded by
        `nodes.MAX_COORDINATOR_ROUNDS` -- see `nodes.coordinator`'s
        docstring for exactly what a retry round does and does not
        recompute.
    needs_knowledge, needs_business : bool
        The coordinator's routing decision -- which specialist branch(es)
        `routing.route_after_coordinator` dispatches to via `Send`. On a
        retry round, `needs_business` is deliberately forced back to
        `False` (see `nodes.coordinator`) so a deterministic business
        lookup that already has a final answer is never redundantly
        re-run.
    is_case_mutation : bool
        Whether the query is a business-write request (reused detection:
        `rag.agent.graph._looks_like_case_mutation_request`). When `True`,
        routing bypasses the knowledge/business-read/merge/critic path
        entirely -- see `routing.route_after_coordinator`.
    selected_specialists : list[str]
        `["knowledge"]`, `["business"]`, or `["knowledge", "business"]` --
        which specialists the *original* (round 0) routing decision
        selected. Read by `evidence_critic` and the evaluation harness;
        never mutated by a retry round.
    knowledge_top_k : int
        Retrieval candidate count for `knowledge_agent`'s
        `pipeline.retrieve()` call. `nodes.DEFAULT_KNOWLEDGE_TOP_K` on the
        first attempt, widened to `nodes.RETRY_KNOWLEDGE_TOP_K` by
        `coordinator` on a critic-requested retry.
    knowledge_evidence : list[EvidenceItem]
        `knowledge_agent`'s retrieved, already-sanitized evidence (see
        `rag.retrieval.pipeline.RetrievalPipeline.retrieve`'s own
        field-redaction/injection-flagging, applied before this state key
        is ever set -- `knowledge_agent` adds no sanitization of its own).
        Overwritten (not accumulated) on a retry -- the latest attempt's
        evidence is what should reach synthesis, not a duplicated union.
    knowledge_status : SpecialistStatus
        `"ok"` (non-empty evidence), `"no_evidence"` (ran, found nothing),
        `"failed"` (the retrieval call itself raised), or `"not_run"`
        (this specialist was never selected). Drives `evidence_critic`'s
        retry decision.
    knowledge_error : str | None
        The failed call's exception class name only -- never
        `str(exc)`/a traceback, matching this codebase's existing
        "shape/timing/outcome fields only" audit-logging discipline (see
        CLAUDE.md's "Logging" section).
    knowledge_tool_calls : list[ToolCallRecord]
        Accumulates (via `operator.add`) across every `knowledge_agent`
        invocation this run, including a retried one -- see this module's
        docstring for why this field, specifically, needs a reducer.
    case_tool_name, case_result, case_found, case_id, requested_new_status,
    pending_action, approval_decision, case_action_outcome
        Owned entirely by the reused business-branch node functions;
        see `nodes.py`'s module docstring. Identical in shape/meaning to
        `langgraph_experiment.state.GraphState`'s same-named fields.
    business_status : SpecialistStatus
        Set by `nodes.business_evaluate_read` (read branch only) from
        `case_found`; stays `"not_run"` on the write branch and on a
        knowledge-only route.
    business_tool_calls : list[ToolCallRecord]
        Same accumulation rationale as `knowledge_tool_calls`, though in
        practice the business branch never re-runs within one thread (see
        `coordinator`'s retry-round docstring), so this is at most a
        single-element list today -- the reducer is there for
        correctness, not because this experiment currently exercises a
        business retry.
    citations : list[Citation]
        Recomputed from scratch by `merge` on every invocation (including
        after a retry) from whatever `knowledge_evidence`/`case_result`
        currently hold -- plain overwrite, deliberately not a reducer (see
        this module's docstring).
    tool_call_log : list[ToolCallRecord]
        `knowledge_tool_calls + business_tool_calls`, recomputed by
        `merge` the same way `citations` is.
    critic_notes : list[str]
        Safe, human-readable diagnostic strings `evidence_critic` appends
        to (manually, reading the prior list -- not a reducer, since only
        `evidence_critic` ever writes this key and a reducer would add no
        value over an explicit read-append-return). Never chain-of-thought
        or raw retrieved/case content -- see `evidence_critic`'s
        docstring for exactly what it does and does not inspect.
    retry_count : int
        How many knowledge-specialist retries `evidence_critic` has
        requested this run. Bounded by `nodes.MAX_KNOWLEDGE_RETRIES`.
    critic_wants_retry : bool
        `evidence_critic`'s routing signal, read by
        `routing.route_after_critic`.
    final_answer : str | None
        The terminal answer text. Set by exactly one of `final_synthesis`
        (knowledge-only/business-read-only/mixed routes) or
        `business_synthesize_write` (the reused, unmodified write-branch
        terminal node).
    termination_reason : str | None
        Why the run ended. See `nodes.final_synthesis`/
        `langgraph_experiment.nodes.synthesize_write_result` for the
        exact value each path sets. `None` only while a run is still in
        progress or paused on the write branch's approval interrupt.
    llm_call_count : int
        How many real `LLM.generate()` calls this run made (0 or 1 --
        `final_synthesis` is the only node in this graph that can call
        one; the write branch's terminal node is purely deterministic
        templating, reused unmodified from `langgraph_experiment.nodes`).
    """

    original_query: str
    thread_id: str
    caller_subject: str
    tenant_id: str | None
    roles: list[str]
    dataset_id: str | None
    workflow_started_at: str

    coordinator_round: int
    needs_knowledge: bool
    needs_business: bool
    is_case_mutation: bool
    selected_specialists: list[str]

    knowledge_top_k: int
    knowledge_evidence: list[EvidenceItem]
    knowledge_status: SpecialistStatus
    knowledge_error: str | None
    knowledge_tool_calls: Annotated[list[ToolCallRecord], operator.add]

    case_id: str | None
    case_tool_name: str | None
    case_result: dict | None
    case_found: bool
    business_status: SpecialistStatus
    business_tool_calls: Annotated[list[ToolCallRecord], operator.add]

    requested_new_status: str | None
    pending_action: PendingCaseAction | None
    approval_decision: dict | None
    case_action_outcome: dict | None

    citations: list[Citation]
    tool_call_log: list[ToolCallRecord]
    critic_notes: list[str]
    retry_count: int
    critic_wants_retry: bool

    final_answer: str | None
    termination_reason: str | None
    llm_call_count: int
