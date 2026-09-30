"""Dependencies the graph's nodes call into, injected once at graph-build time.

Reuses `rag.factory`/`rag.config`/`rag.mcp.business.store`/
`rag.retrieval.pipeline` wholesale; this module adds no retrieval,
generation, or business-rule logic of its own -- only wiring, mirroring
`rag.agent.graph.run_agent`'s own injected-singleton parameters
(`pipeline`, `vectorstore`, `embedder`, `llm`).

`GraphDeps.get_customer_case_fn`/`get_case_status_fn`/`update_case_status_fn`
default to the real `rag.mcp.business.store` functions but are ordinary
fields, so a test can substitute a fake (or one that raises
`TransientCaseStoreError` a bounded number of times, to exercise the
`execute_case_read_tool` node's `RetryPolicy`) without touching graph
wiring at all.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from langgraph_experiment.idempotency import ActionLedger
from rag.api.auth import VerifiedIdentity
from rag.config import AppConfig, load_config
from rag.factory import build_llm
from rag.generation.base import LLM
from rag.mcp.business import store as case_store
from rag.mcp.business.schemas import CaseActionOutcome, CaseApproval, CaseStatusResult, CustomerCase
from rag.observability import tracing
from rag.prompts.loader import PromptTemplate, load_prompt_template_from_config
from rag.retrieval.pipeline import RetrievalPipeline

#: Default approval wait window and overall per-workflow ceiling. Neither
#: is read from `config/default.yaml` (they're specific to this
#: experiment, not a production RAG setting) -- plain module constants,
#: overridable per `GraphDeps` instance.
DEFAULT_APPROVAL_TIMEOUT_SECONDS = 15 * 60
DEFAULT_MAX_WORKFLOW_DURATION_SECONDS = 60 * 60


class TransientCaseStoreError(Exception):
    """A retryable failure reaching the (synthetic) case-store backend.

    Never raised by `rag.mcp.business.store` itself -- that module is
    in-memory and has no transient-failure mode. Exists so
    `execute_case_read_tool`'s `RetryPolicy` (see `graph.build_graph`)
    and the "failure recovery" test scenario have a realistic exception
    class to inject via a fake `get_customer_case_fn`, standing in for
    a flaky network call a real remote MCP business backend could have.
    """


GetCustomerCaseFn = Callable[[str, VerifiedIdentity | None, list[str]], CustomerCase | None]
GetCaseStatusFn = Callable[[str, VerifiedIdentity | None, list[str]], CaseStatusResult | None]
UpdateCaseStatusFn = Callable[
    [str, str, VerifiedIdentity | None, list[str], list[CaseApproval]], CaseActionOutcome | None
]


@dataclass
class GraphDeps:
    """Everything the graph's nodes need. Built once per graph, shared by every node.

    Attributes
    ----------
    config : AppConfig
        The loaded application config; only `security.authorization.
        cross_tenant_support_roles` and `mcp.business_actions.
        approval_roles` are actually read directly (the rest flows
        through `pipeline`/`llm`, already built from it).
    pipeline : RetrievalPipeline | None
        Built for the read-only RAG branch. `None` skips that branch's
        Postgres/embedding-model dependency entirely (see
        `build_default_deps`'s `with_retrieval` flag) -- the case-store
        branch, including the whole interrupt/approval demo, needs
        neither Postgres nor Ollama, since `rag.mcp.business.store` is a
        self-contained in-memory backend.
    llm, rag_prompt_template
        Reused for the read-only RAG branch's synthesis step, built the
        same way `RetrievalPipeline.__init__` builds its own (private)
        copies -- kept here instead of reaching into `pipeline`'s private
        attributes, since `synthesize_rag` needs `generate()`/`render()`
        directly rather than the whole `pipeline.answer()` convenience
        method (the point of the experiment's node split is one node per
        pipeline stage; see README.md).
    approval_roles, cross_tenant_support_roles
        Copied out of `config` at build time so tests can override them
        without constructing a full `AppConfig`.
    get_customer_case_fn, get_case_status_fn, update_case_status_fn
        Default to the real `rag.mcp.business.store` functions.
    action_ledger : ActionLedger | None
        The Postgres-backed idempotency ledger `execute_write_action`
        guards every mutation attempt through (see `idempotency.py`).
        `None` disables it entirely -- `execute_write_action` then calls
        `update_case_status_fn` directly with no ledger involvement,
        which is what every test from the original (pre-production-
        hardening) phase still does, needing no Postgres at all.
    approval_timeout_seconds : int
        How long a paused write action waits for a decision before
        `wait_for_approval` refuses to honor any resume for it, checked
        server-side against `PendingCaseAction.expires_at`.
    max_workflow_duration_seconds : int
        A separate, coarser ceiling on total elapsed time since
        `workflow_started_at`, independent of the approval-specific
        window above -- see `state.GraphState.workflow_started_at`'s
        docstring for why these are two distinct settings even though
        this graph's topology only ever lets either one be observed at
        the same single interrupt point.
    """

    config: AppConfig
    pipeline: RetrievalPipeline | None = None
    llm: LLM | None = None
    rag_prompt_template: PromptTemplate | None = None
    approval_roles: tuple[str, ...] = ("case_status_approver",)
    cross_tenant_support_roles: tuple[str, ...] = ("techfusion_support",)
    get_customer_case_fn: GetCustomerCaseFn = field(default=case_store.get_customer_case)
    get_case_status_fn: GetCaseStatusFn = field(default=case_store.get_case_status)
    update_case_status_fn: UpdateCaseStatusFn = field(default=case_store.update_case_status)
    action_ledger: ActionLedger | None = None
    approval_timeout_seconds: int = DEFAULT_APPROVAL_TIMEOUT_SECONDS
    max_workflow_duration_seconds: int = DEFAULT_MAX_WORKFLOW_DURATION_SECONDS


def build_default_deps(
    config: AppConfig | None = None, *, with_retrieval: bool = True, with_ledger: bool = True
) -> GraphDeps:
    """Build production-shaped deps from `config/default.yaml` and `.env`.

    Parameters
    ----------
    config : AppConfig | None, optional
        Loaded via `rag.config.load_config()` if omitted.
    with_retrieval : bool, optional
        When `False`, skips building `pipeline`/`llm`/`rag_prompt_template`
        (and therefore any Postgres/Ollama connection attempt) -- for a
        demo run that only exercises the case-store branch, which has no
        external dependency at all. `True` by default so `retrieve()`/
        `synthesize_rag` work out of the box for a caller that does have
        `make up` and Ollama running.
    with_ledger : bool, optional
        When `True` (the default), builds a real `ActionLedger` against
        `config.database_url()` and calls `ensure_schema()` once -- needs
        the same Postgres `with_retrieval` already assumes. `False` skips
        it (`GraphDeps.action_ledger=None`), for a demo run with no
        Postgres available at all -- the write-action branch still works,
        just without the crash-recovery idempotency guarantee (falling
        back to `update_case_status`'s own natural idempotency alone).
    also configures OpenTelemetry tracing once, from `config.
    observability.tracing` (idempotent, a true no-op unless that config
    section enables it -- see `rag.observability.tracing.
    configure_tracing`).

    Returns
    -------
    GraphDeps
        Ready to pass to `langgraph_experiment.graph.build_graph`.
    """
    resolved_config = config or load_config()
    tracing.configure_tracing(resolved_config)
    pipeline = RetrievalPipeline(resolved_config) if with_retrieval else None
    llm = build_llm(resolved_config) if with_retrieval else None
    prompt_template = load_prompt_template_from_config(resolved_config) if with_retrieval else None
    action_ledger: ActionLedger | None = None
    if with_ledger:
        action_ledger = ActionLedger(resolved_config.database_url())
        action_ledger.ensure_schema()
    return GraphDeps(
        config=resolved_config,
        pipeline=pipeline,
        llm=llm,
        rag_prompt_template=prompt_template,
        approval_roles=tuple(resolved_config.mcp.business_actions.approval_roles),
        cross_tenant_support_roles=tuple(
            resolved_config.security.authorization.cross_tenant_support_roles
        ),
        action_ledger=action_ledger,
    )
