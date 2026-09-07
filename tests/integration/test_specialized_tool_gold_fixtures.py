"""Deterministic proof for every scenario in `data/eval/specialized_tool_gold.jsonl`.

This file exists to answer one question independently of any live LLM
run: for each specialized-tool gold example, does the underlying tool
actually behave the way the gold row's `expected_*` fields claim, when
dispatched directly with that row's exact tenant/role/case_id/source?
This is "tool reachability and correctness," never "did the model choose
to call it" -- see `rag.eval.run_agent_eval`'s `per_tool_reachability`/
`mcp_authorization`/`case_action_outcome_accuracy` metrics for the
live-LLM-selection side of that distinction.

Two independent groups:

- MCP business-tool scenarios (the eight `specialized_mcp_case_read`/
  `specialized_mcp_case_action` gold rows): call `rag.mcp.business.store`
  directly, no Postgres/Ollama needed. `_reset_case_store` restores the
  shared in-memory `_SYNTHETIC_CASES` dict after every test, matching
  `test_mcp_business_case_actions.py`'s established pattern -- this suite
  adds no new mutation risk beyond what that file already accepts.
- Local specialized-tool scenarios (the six `specialized_get_document`/
  `specialized_get_latest_document`/`specialized_get_related_context`
  gold rows): call `rag.agent.tools` directly against the real, already
  -ingested `techfusion` dataset (`require_postgres`; no re-ingestion, no
  mutation -- these are read-only lookups against existing corpus
  documents by their real, stable `source` path). Confirms the exact
  documents/relationships these gold questions depend on actually exist
  and resolve the way the gold row's `agentic_rationale` claims, before
  any live-model run is trusted to get the same answer.
"""

from __future__ import annotations

import copy

import pytest

from rag.agent.tool_schemas import GetDocumentArgs, GetLatestDocumentArgs, GetRelatedContextArgs
from rag.agent.tools import get_document, get_latest_document, get_related_context
from rag.api.auth import VerifiedIdentity
from rag.eval.gold_schema import source_matches_relevant
from rag.factory import build_vectorstore
from rag.mcp.business import store as business_store
from rag.mcp.business.schemas import CaseApproval
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.pipeline import RetrievalPipeline
from rag.vectorstore.base import VectorStore


def _resolve_source(vectorstore: VectorStore, dataset_id: str, relative_suffix: str) -> str:
    """Resolve a gold-style relative path to this environment's real, exact stored `source`.

    `get_chunks_by_source` matches `source` exactly, and the stored value
    embeds whatever OS path separator ingestion ran under (this dev
    corpus was ingested natively on Windows, so it's backslash-separated
    on disk; a container-ingested corpus would be forward-slash). Reusing
    `source_matches_relevant`'s existing path-suffix rule keeps this test
    file portable across both, matching exactly what gold's own
    `relevant_documents` fields already rely on.
    """
    for source in vectorstore.list_document_sources(dataset_id):
        if source_matches_relevant(source, relative_suffix):
            return source
    raise AssertionError(f"no document with suffix {relative_suffix!r} found in {dataset_id!r}")


_SUPPORT_ROLES = ["techfusion_support"]


class _NoOpEmbedder:
    """Placeholder embedder: every document fetched here fits under max_chunks unselected."""

    def embed_query(self, text: str) -> list[float]:
        return [0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


def _identity(tenant_id: str, roles: list[str]) -> VerifiedIdentity:
    return VerifiedIdentity(subject="gold-fixture-check", tenant_id=tenant_id, roles=roles)


def _secure_config(config):
    secure = config.model_copy(deep=True)
    secure.security.authorization.enabled = True
    return secure


# --- MCP business-tool scenarios ------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_case_store():
    original = copy.deepcopy(business_store._SYNTHETIC_CASES)
    yield
    business_store._SYNTHETIC_CASES.clear()
    business_store._SYNTHETIC_CASES.update(original)


def test_gold_case_status_success_case_1001_tenant_alpha_operator():
    """Row 7: get_case_status(CASE-1001) as tenant_alpha_operator is authorized and correct."""
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])
    status = business_store.get_case_status("CASE-1001", identity, _SUPPORT_ROLES)
    assert status is not None
    assert status.status == "in_progress"
    assert status.priority == "high"


def test_gold_customer_case_success_case_2002_cross_tenant_support():
    """Row 8: get_customer_case(CASE-2002) as internal_techfusion/techfusion_support succeeds.

    Also proves the description text the gold expected_answer relies on
    (the cross-case link to CASE-1001) is really present.
    """
    identity = _identity("internal_techfusion", ["techfusion_support"])
    case = business_store.get_customer_case("CASE-2002", identity, _SUPPORT_ROLES)
    assert case is not None
    assert case.tenant_id == "tenant_beta"
    assert "CASE-1001" in case.description


def test_gold_case_status_denied_case_2001_cross_tenant_no_support_role():
    """Row 9: get_case_status(CASE-2001) as plain tenant_alpha_operator is denied (None)."""
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])
    assert business_store.get_case_status("CASE-2001", identity, _SUPPORT_ROLES) is None


def test_gold_customer_case_denied_case_1002_same_tenant_wrong_role():
    """Row 10: get_customer_case(CASE-1002) as tenant_alpha_operator is denied (admin-only)."""
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])
    assert business_store.get_customer_case("CASE-1002", identity, _SUPPORT_ROLES) is None


def test_gold_update_case_status_approval_required_case_2001_tenant_beta_operator():
    """Row 11: CASE-2001 resolved -> closed as tenant_beta_operator needs approval, not mutated."""
    identity = _identity("tenant_beta", ["tenant_beta_operator"])
    outcome = business_store.update_case_status("CASE-2001", "closed", identity, _SUPPORT_ROLES)
    assert outcome is not None
    assert outcome.outcome == "approval_required"
    assert business_store._SYNTHETIC_CASES["CASE-2001"].status == "resolved"


def test_gold_update_case_status_approval_required_becomes_executed_with_approval():
    """The same row 11 scenario, but with a matching approval attached, actually executes.

    Not itself a gold row (the live-eval subset never supplies
    case_approvals, so it can only ever observe approval_required for
    this scenario) -- proves the approval path this scenario is
    contrasted against is real, not just assumed.
    """
    identity = _identity("tenant_beta", ["tenant_beta_operator"])
    approvals = [CaseApproval(case_id="CASE-2001", new_status="closed")]
    outcome = business_store.update_case_status(
        "CASE-2001", "closed", identity, _SUPPORT_ROLES, approvals
    )
    assert outcome is not None
    assert outcome.outcome == "executed"
    assert business_store._SYNTHETIC_CASES["CASE-2001"].status == "closed"


def test_gold_update_case_status_invalid_transition_case_1003_tenant_alpha_operator():
    """Row 12: closed -> open on CASE-1003 as tenant_alpha_operator is rejected, no mutation."""
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])
    outcome = business_store.update_case_status("CASE-1003", "open", identity, _SUPPORT_ROLES)
    assert outcome is not None
    assert outcome.outcome == "invalid_transition"
    assert business_store._SYNTHETIC_CASES["CASE-1003"].status == "closed"


def test_gold_update_case_status_already_in_status_case_1001_tenant_alpha_operator():
    """Row 13: in_progress -> in_progress on CASE-1001 is a deterministic no-op."""
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])
    outcome = business_store.update_case_status(
        "CASE-1001", "in_progress", identity, _SUPPORT_ROLES
    )
    assert outcome is not None
    assert outcome.outcome == "already_in_status"
    assert business_store._SYNTHETIC_CASES["CASE-1001"].status == "in_progress"


def test_gold_update_case_status_denied_case_2001_cross_tenant_tenant_alpha_operator():
    """Row 14: resolved -> closed on CASE-2001 as tenant_alpha_operator is denied outright (None).

    Contrast with the approval_required test above: same case, same
    target status, but this caller has no access to CASE-2001 at all, so
    the denial must happen before the transition table is ever
    consulted.
    """
    identity = _identity("tenant_alpha", ["tenant_alpha_operator"])
    outcome = business_store.update_case_status("CASE-2001", "closed", identity, _SUPPORT_ROLES)
    assert outcome is None
    assert business_store._SYNTHETIC_CASES["CASE-2001"].status == "resolved"


# --- Local specialized-tool scenarios (real, already-ingested techfusion corpus) --------


def test_gold_get_document_postgres_recovery_spans_troubleshooting_and_closure(
    require_postgres, config
):
    """Row 1: get_document on the PostgreSQL Recovery runbook covers both distant sections."""
    secure = _secure_config(config)
    vectorstore = build_vectorstore(secure)
    pipeline = RetrievalPipeline(secure, vectorstore=vectorstore)
    source = _resolve_source(
        vectorstore, "techfusion", "knowledge_base/runbooks/postgres-recovery.md"
    )

    chunks = get_document(
        GetDocumentArgs(source=source),
        pipeline,
        vectorstore,
        "techfusion",
        "WAL segments missing closure recovery point",
        _NoOpEmbedder(),
        None,
        max_chunks=50,
        max_chunks_hard_ceiling=50,
    )
    text = " ".join(c.content for c in chunks)
    assert "Missing WAL segments" in text
    assert "Reapply deletion tombstones" in text


def test_gold_get_document_access_control_policy_spans_three_time_windows(require_postgres, config):
    """Row 2: get_document on the Access Control Policy covers all three time-bounded windows."""
    secure = _secure_config(config)
    vectorstore = build_vectorstore(secure)
    pipeline = RetrievalPipeline(secure, vectorstore=vectorstore)
    source = _resolve_source(
        vectorstore, "techfusion", "knowledge_base/security/access-control-policy.md"
    )

    chunks = get_document(
        GetDocumentArgs(source=source),
        pipeline,
        vectorstore,
        "techfusion",
        "time-bounded access windows just-in-time break-glass impersonation",
        _NoOpEmbedder(),
        None,
        max_chunks=50,
        max_chunks_hard_ceiling=50,
    )
    text = " ".join(c.content for c in chunks)
    assert "normally limited to one hour" in text
    assert "expires after one hour" in text
    assert "maximum 30-minute session" in text


def test_gold_get_latest_document_incident_response_v1_redirects_to_v2(require_postgres, config):
    """Row 3: naming incident-response-v1.md directly still resolves to v2's current values."""
    secure = _secure_config(config)
    vectorstore = build_vectorstore(secure)
    pipeline = RetrievalPipeline(secure, vectorstore=vectorstore)
    auth = AuthorizationContext(tenant_id="internal_techfusion", roles=["security_analyst"])
    v1_source = _resolve_source(
        vectorstore,
        "techfusion",
        "knowledge_base/security_evaluation/internal_techfusion/incident-response-v1.md",
    )

    chunks = get_latest_document(
        GetLatestDocumentArgs(source=v1_source),
        pipeline,
        vectorstore,
        "techfusion",
        "SEV-1 acknowledgement incident commander",
        _NoOpEmbedder(),
        auth,
        max_chunks=50,
        max_chunks_hard_ceiling=50,
    )
    text = " ".join(c.content for c in chunks)
    assert "10 minutes" in text
    assert "Security duty manager" in text
    assert "15 minutes" not in text
    assert "Platform on-call" not in text


def test_gold_get_latest_document_retention_policy_v1_redirects_to_v2(require_postgres, config):
    """Row 4: naming retention-policy-v1.md directly still resolves to v2's 90-day value."""
    secure = _secure_config(config)
    vectorstore = build_vectorstore(secure)
    pipeline = RetrievalPipeline(secure, vectorstore=vectorstore)
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["tenant_alpha_viewer"])
    v1_source = _resolve_source(
        vectorstore,
        "techfusion",
        "knowledge_base/security_evaluation/tenant_alpha/retention-policy-v1.md",
    )

    chunks = get_latest_document(
        GetLatestDocumentArgs(source=v1_source),
        pipeline,
        vectorstore,
        "techfusion",
        "processed production document retention period",
        _NoOpEmbedder(),
        auth,
        max_chunks=50,
        max_chunks_hard_ceiling=50,
    )
    text = " ".join(c.content for c in chunks)
    assert "90 days" in text


def test_gold_get_related_context_ocr_retry_config_reaches_lock_ttl_explanation(
    require_postgres, config
):
    """Row 5: get_related_context on the OCR retry JSON config reaches its parent prose."""
    secure = _secure_config(config)
    vectorstore = build_vectorstore(secure)
    pipeline = RetrievalPipeline(secure, vectorstore=vectorstore)
    source = _resolve_source(
        vectorstore, "techfusion", "knowledge_base/architecture/document-processing-sequence.md"
    )

    doc_chunks = vectorstore.get_chunks_by_source(source, "techfusion", auth=None)
    config_chunk = next(c for c in doc_chunks if "retry_lock_ttl_seconds" in c.content)

    related = get_related_context(
        GetRelatedContextArgs(chunk_id=config_chunk.metadata.chunk_id),
        pipeline,
        vectorstore,
        None,
        dataset_id="techfusion",
    )
    text = " ".join(c.content for c in related)
    assert "Lock TTL is set independently" in text
    assert "capped at 480" in text


def test_gold_get_related_context_latency_budget_config_reaches_rollout_rationale(
    require_postgres, config
):
    """Row 6: get_related_context on the latency-budget config reaches the rollout rationale."""
    secure = _secure_config(config)
    vectorstore = build_vectorstore(secure)
    pipeline = RetrievalPipeline(secure, vectorstore=vectorstore)
    source = _resolve_source(
        vectorstore, "techfusion", "knowledge_base/engineering/retrieval-performance-analysis.md"
    )

    doc_chunks = vectorstore.get_chunks_by_source(source, "techfusion", auth=None)
    config_chunk = next(c for c in doc_chunks if "p95_budget_ms" in c.content)

    related = get_related_context(
        GetRelatedContextArgs(chunk_id=config_chunk.metadata.chunk_id),
        pipeline,
        vectorstore,
        None,
        dataset_id="techfusion",
    )
    text = " ".join(c.content for c in related)
    assert "improved relevance consistently" in text
