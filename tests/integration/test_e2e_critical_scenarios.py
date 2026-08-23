"""End-to-end critical-path proofs through the real HTTP API.

Real Postgres, real Ollama, the real `rag.api.main.app`. No fakes, no
mocks, no isolated test app. Each test tells one complete story a real
caller would experience: ingest -> authenticate (where relevant) -> call
the public endpoint -> assert on the exact response an external caller
actually receives.

Deliberately distinct from the narrower integration tests elsewhere in this
directory (`test_query_pipeline.py`, `test_agent_end_to_end.py`,
`test_authorization_isolation.py`, `test_field_level_redaction.py`, ...),
which mostly call `RetrievalPipeline`/`run_agent()` directly to prove one
mechanism in isolation. These tests instead exercise the full request
lifecycle through `TestClient(app)`: JWT header parsing, request
validation, the real dependency-injection wiring, and the exact JSON/SSE
shape a caller gets back.

Where a scenario needs a different config than whatever the process-wide
default singletons were first built from (a stronger model for agent
JSON-schema reliability, or a security toggle that's off by default),
`app.dependency_overrides` swaps the specific FastAPI dependency the route
actually declares. Overriding `get_config` alone is not enough for
`get_vectorstore`/`get_embedder`/`get_llm`/`get_retrieval_pipeline`: those
are separate `@lru_cache`d singletons that call `get_config()` as a plain
Python function, not through FastAPI's own dependency resolution (see
`api/deps.py`'s module docstring), so every `Depends()` target a given
route declares must be overridden individually, matching the pattern
`test_api_field_redaction.py`/`test_agent_query_stream.py` already
established.

Assertions on real-model free text stay deliberately robust to run-to-run
LLM variability, matching `test_agent_end_to_end.py`'s documented
philosophy: never assert exact wording; assert structural/security-relevant
properties instead (status codes, which sources/citations came back,
whether a secret literal appears anywhere in the raw response body).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

import jwt
from fastapi.testclient import TestClient

from rag.api.deps import get_config, get_embedder, get_llm, get_retrieval_pipeline, get_vectorstore
from rag.api.main import app
from rag.eval.run_eval import _looks_like_refusal
from rag.factory import build_embedder, build_llm, build_vectorstore
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.pipeline import RetrievalPipeline

_JWT_SECRET = "e2e-integration-test-only-not-a-real-secret-value"

_VALID_TERMINATIONS = {
    "synthesized",
    "max_steps",
    "max_retrieval_attempts",
    "max_tool_calls",
    "insufficient_evidence",
}

_AGENT_OVERRIDE_DEPS = (get_config, get_retrieval_pipeline, get_vectorstore, get_embedder, get_llm)


def _write_doc(tmp_path: Path, name: str, frontmatter: dict[str, Any], body: str) -> Path:
    """Write a synthetic Markdown file, with an optional YAML front-matter block."""
    lines: list[str] = []
    if frontmatter:
        lines.append("---")
        for key, value in frontmatter.items():
            if isinstance(value, list):
                lines.append(f"{key}:")
                lines.extend(f'  - "{item}"' for item in value)
            else:
                lines.append(f'{key}: "{value}"')
        lines.append("---")
        lines.append("")
    lines.append(body)
    path = tmp_path / name
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _token(tenant_id: str, roles: list[str]) -> str:
    now = int(time.time())
    claims = {
        "sub": "pytest-e2e",
        "tenant_id": tenant_id,
        "roles": roles,
        "iat": now,
        "exp": now + 3600,
    }
    return jwt.encode(claims, _JWT_SECRET, algorithm="HS256")


def _secure_config(config, *, field_redaction: bool = False):
    """Return a copy of `config` with JWT auth + tenant/role authorization on (default model)."""
    secure = config.model_copy(deep=True)
    secure.security.auth.enabled = True
    secure.security.auth.jwt.secret_env_var = "JWT_HS256_SECRET"
    secure.security.authorization.enabled = True
    secure.security.field_redaction.enabled = field_redaction
    return secure


def _agentic_config(config):
    """Return a copy of `config` with a stronger model for agent JSON-schema reliability.

    Matches the model choice PROJECT_JOURNAL records as substantially more
    reliable than the default 1.5b for classification/tool-selection JSON
    (same choice `test_agent_end_to_end.py`/`test_agent_query_stream.py` make).
    """
    agentic = config.model_copy(deep=True)
    agentic.generation.model_name = "qwen2.5:3b"
    return agentic


def _wire_agent_dependencies(agentic_config):
    """Build fresh real singletons for `agentic_config` and wire them onto the real app.

    Returns (vectorstore, embedder, pipeline). Caller must
    `_clear_overrides()` in a `finally` block.
    """
    vectorstore = build_vectorstore(agentic_config)
    embedder = build_embedder(agentic_config)
    llm = build_llm(agentic_config)
    pipeline = RetrievalPipeline(
        agentic_config, vectorstore=vectorstore, embedder=embedder, llm=llm
    )
    app.dependency_overrides[get_config] = lambda: agentic_config
    app.dependency_overrides[get_retrieval_pipeline] = lambda: pipeline
    app.dependency_overrides[get_vectorstore] = lambda: vectorstore
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_llm] = lambda: llm
    return vectorstore, embedder, pipeline


def _clear_overrides(deps) -> None:
    for dep in deps:
        app.dependency_overrides.pop(dep, None)


def _parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    event_type = None
    for line in body.splitlines():
        if line.startswith("event: "):
            event_type = line[len("event: ") :]
        elif line.startswith("data: ") and event_type is not None:
            events.append((event_type, json.loads(line[len("data: ") :])))
            event_type = None
    return events


# ---------------------------------------------------------------------------
# 1. Classic RAG: ingest a known document -> POST /query -> grounded answer + sources.
# ---------------------------------------------------------------------------


def test_classic_rag_ingest_then_query_returns_grounded_answer_with_sources(
    require_postgres, require_ollama, config, tmp_path: Path
):
    """The system's core promise, end to end, with the shipped default config.

    No auth, no agent, no security toggles. This is exactly what a fresh clone
    gets after `make up` + ingest. Proves the answer is both non-empty and
    grounded (contains a fact only present in the ingested document) and
    that the source citation points back at the real ingested file.
    """
    ns = f"pytest-e2e-classic-{uuid.uuid4()}"
    doc = _write_doc(
        tmp_path,
        "oncall-escalation-policy.md",
        {},
        "# On-Call Escalation Policy\n\n"
        "A Sev1 incident must be acknowledged within 5 minutes. If unacknowledged, "
        "PagerDuty automatically escalates to the secondary on-call engineer, then "
        "to the engineering manager after a further 10 minutes.",
    )
    ingestion = IngestionPipeline(config)
    ingestion.ingest_file(doc, ns)

    try:
        client = TestClient(app)
        response = client.post(
            "/query",
            json={
                "query": "How quickly must a Sev1 incident be acknowledged, and what "
                "system handles escalation?",
                "filters": {"dataset_id": ns},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        # A distinctive, unlikely-to-be-paraphrased proper noun from the source --
        # a robust grounding check that doesn't depend on the model's exact wording.
        assert "pagerduty" in body["answer"].lower()
        assert body["sources"], "expected at least one source citation"
        assert all("oncall-escalation-policy" in s["source"] for s in body["sources"])
        assert body["retrieval_ms"] > 0
        assert body["total_ms"] >= body["retrieval_ms"]
    finally:
        ingestion._vectorstore.delete_dataset(ns)


# ---------------------------------------------------------------------------
# 2. Agentic RAG: query -> classification -> tool execution -> synthesis -> citations.
# ---------------------------------------------------------------------------


def test_agentic_rag_two_hop_query_produces_citations_grounded_in_both_documents(
    require_postgres, require_ollama, config, tmp_path: Path
):
    """A genuinely two-hop question through POST /agent/query.

    Answering it requires combining a fact from one document (which region
    a dataset is replicated to) with a fact from a second, otherwise
    unrelated document (that region's maintenance window), the kind of
    "more than one dependent lookup" question `agent_classify_v2.yaml`
    documents as its complex-routing bar. Real-LLM answer wording isn't
    asserted; citation grounding (every returned source is one of the two
    real ingested documents, never a hallucinated path) is.
    """
    ns = f"pytest-e2e-agentic-{uuid.uuid4()}"
    agentic = _agentic_config(config)
    vectorstore, embedder, _pipeline = _wire_agent_dependencies(agentic)
    ingestion = IngestionPipeline(agentic, vectorstore=vectorstore, embedder=embedder)

    residency_doc = _write_doc(
        tmp_path,
        "data-residency-map.md",
        {},
        "# Data Residency Map\n\n"
        "The `analytics_warehouse` dataset is replicated to the eu-west-2 region "
        "for disaster recovery purposes.",
    )
    maintenance_doc = _write_doc(
        tmp_path,
        "region-maintenance-windows.md",
        {},
        "# Region Maintenance Windows\n\n"
        "The eu-west-2 region's scheduled maintenance window is Sundays 02:00-04:00 UTC.",
    )
    ingestion.ingest_file(residency_doc, ns)
    ingestion.ingest_file(maintenance_doc, ns)
    known_sources = {"data-residency-map", "region-maintenance-windows"}

    try:
        client = TestClient(app)
        response = client.post(
            "/agent/query",
            json={
                "query": "Which maintenance window applies to the region where the "
                "analytics_warehouse dataset is replicated?",
                "filters": {"dataset_id": ns},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["route"] in ("classic_rag", "agent")
        assert body["termination_reason"] in _VALID_TERMINATIONS
        assert body["answer"]
        assert body["sources"], "expected at least one citation"
        # Never a hallucinated citation: every source traces back to a real
        # ingested document, not a guessed or fabricated path.
        for source in body["sources"]:
            assert any(known in source["source"] for known in known_sources)
        if body["route"] == "agent":
            assert body["steps"] >= 1
            assert body["tool_calls"], "agent route should have dispatched at least one tool call"
    finally:
        vectorstore.delete_dataset(ns)
        _clear_overrides(_AGENT_OVERRIDE_DEPS)


# ---------------------------------------------------------------------------
# 3. Authorization: JWT tenant A -> query -> cannot retrieve tenant B evidence.
# ---------------------------------------------------------------------------


def test_authorization_jwt_tenant_cannot_retrieve_other_tenants_evidence(
    require_postgres, require_ollama, config, tmp_path: Path, monkeypatch
):
    """A verified tenant_alpha caller retrieves only tenant_alpha's document via POST /query.

    Two tenant-scoped documents exist side by side in the same dataset;
    document-level ACL (`security.authorization.enabled`) must keep them
    fully separate through the real JWT -> AuthorizationContext -> SQL
    predicate path, not just when calling internals directly.
    """
    monkeypatch.setenv("JWT_HS256_SECRET", _JWT_SECRET)
    ns = f"pytest-e2e-authz-{uuid.uuid4()}"
    secure = _secure_config(config)
    ingestion = IngestionPipeline(secure)

    alpha_doc = _write_doc(
        tmp_path,
        "tenant-alpha-runbook.md",
        {"tenant_id": "tenant_alpha", "allowed_roles": ["tenant_alpha_operator"]},
        "The Tenant Alpha maintenance contact is Priya Alpha-Ops, reachable at "
        "internal extension 4471.",
    )
    beta_doc = _write_doc(
        tmp_path,
        "tenant-beta-runbook.md",
        {"tenant_id": "tenant_beta", "allowed_roles": ["tenant_beta_operator"]},
        "The Tenant Beta maintenance contact is Sam Beta-Ops, reachable at "
        "internal extension 8823.",
    )
    ingestion.ingest_file(alpha_doc, ns)
    ingestion.ingest_file(beta_doc, ns)

    app.dependency_overrides[get_config] = lambda: secure
    app.dependency_overrides[get_retrieval_pipeline] = lambda: RetrievalPipeline(secure)
    try:
        client = TestClient(app)
        token = _token("tenant_alpha", ["tenant_alpha_operator"])
        response = client.post(
            "/query",
            json={
                "query": "Who is the maintenance contact and what is their extension?",
                "filters": {"dataset_id": ns},
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        body = response.json()
        # Structural, always-true regardless of exact wording: tenant_beta's
        # document must never appear in the citation list at all.
        assert all("tenant-beta-runbook" not in s["source"] for s in body["sources"])
        # And its content must never leak anywhere in the raw response body,
        # not just be missing from the (already narrow) sources contract.
        assert "Sam Beta-Ops" not in response.text
        assert "8823" not in response.text
    finally:
        app.dependency_overrides.pop(get_config, None)
        app.dependency_overrides.pop(get_retrieval_pipeline, None)
        ingestion._vectorstore.delete_dataset(ns)


# ---------------------------------------------------------------------------
# 4. Security/redaction: restricted field -> complete request -> secret never
#    appears in the final answer.
# ---------------------------------------------------------------------------


def test_field_redaction_secret_never_reaches_the_final_answer_for_an_unauthorized_role(
    require_postgres, require_ollama, config, tmp_path: Path, monkeypatch
):
    """A document both roles may access, but one field only an admin role may see.

    Distinct from the authorization scenario above: document-level ACL
    admits the operator role (same document, same tenant), but the
    specific credential field is redacted before it ever reaches the
    prompt. Also proves the deterministic marker-sanitization backstop:
    not just that the raw secret is absent, but that the internal
    `[REDACTED:SENSITIVE_FIELD]` marker text never leaks into the
    caller-facing answer either.
    """
    monkeypatch.setenv("JWT_HS256_SECRET", _JWT_SECRET)
    ns = f"pytest-e2e-redaction-{uuid.uuid4()}"
    secure = _secure_config(config, field_redaction=True)
    ingestion = IngestionPipeline(secure)

    secret = "SYNTHETIC_ONLY_E2E_VAULT_9Q2R"
    doc = _write_doc(
        tmp_path,
        "vault-access-runbook.md",
        {
            "tenant_id": "tenant_alpha",
            "allowed_roles": ["tenant_alpha_operator", "tenant_alpha_admin"],
        },
        f"The synthetic vault unlock code is {secret}.",
    )
    ingestion.ingest_file(doc, ns)

    app.dependency_overrides[get_config] = lambda: secure
    app.dependency_overrides[get_retrieval_pipeline] = lambda: RetrievalPipeline(secure)
    try:
        client = TestClient(app)

        operator_token = _token("tenant_alpha", ["tenant_alpha_operator"])
        operator_response = client.post(
            "/query",
            json={
                "query": "What is the synthetic vault unlock code?",
                "filters": {"dataset_id": ns},
            },
            headers={"Authorization": f"Bearer {operator_token}"},
        )
        assert operator_response.status_code == 200
        assert secret not in operator_response.text
        assert "[REDACTED:SENSITIVE_FIELD]" not in operator_response.json()["answer"]

        # Document-level ACL still admits the admin role to the document itself
        # Retrieval isn't blocked outright; only the field is policy-gated.
        admin_token = _token("tenant_alpha", ["tenant_alpha_admin"])
        admin_response = client.post(
            "/query",
            json={
                "query": "What is the synthetic vault unlock code?",
                "filters": {"dataset_id": ns},
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert admin_response.status_code == 200
        assert admin_response.json()["sources"], "admin role must not be blocked from the document"
    finally:
        app.dependency_overrides.pop(get_config, None)
        app.dependency_overrides.pop(get_retrieval_pipeline, None)
        ingestion._vectorstore.delete_dataset(ns)


# ---------------------------------------------------------------------------
# 5. Streaming: POST /agent/query/stream -> valid event sequence -> final response.
# ---------------------------------------------------------------------------


def test_streaming_agent_query_emits_safe_events_ending_in_a_grounded_final_response(
    require_postgres, require_ollama, config, tmp_path: Path
):
    """A real streamed agent run through the real app: safe events, then a grounded answer.

    Complementary to `test_agent_query_stream.py` (which uses a stripped-
    down isolated app mounting only `agent_stream.router`): this goes
    through the real `rag.api.main.app`, and additionally checks that the
    terminal event's citations trace back to the real ingested document.
    """
    ns = f"pytest-e2e-stream-{uuid.uuid4()}"
    agentic = _agentic_config(config)
    vectorstore, embedder, _pipeline = _wire_agent_dependencies(agentic)
    ingestion = IngestionPipeline(agentic, vectorstore=vectorstore, embedder=embedder)

    doc = _write_doc(
        tmp_path,
        "cache-invalidation-runbook.md",
        {},
        "# Cache Invalidation Runbook\n\n"
        "To force-invalidate the edge cache for a stale asset, run "
        "`cache purge --key <asset-id>` and confirm propagation completes within 2 minutes.",
    )
    ingestion.ingest_file(doc, ns)

    try:
        client = TestClient(app)
        with client.stream(
            "POST",
            "/agent/query/stream",
            json={
                "query": "How do I force-invalidate the edge cache for a stale asset?",
                "filters": {"dataset_id": ns},
            },
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

        events = _parse_sse(body)
        assert events, "expected at least one SSE event"

        safe_types = {
            "query_received",
            "route_selected",
            "decomposition_started",
            "decomposition_completed",
            "tool_selected",
            "tool_started",
            "tool_completed",
            "evidence_evaluated",
            "retry_started",
            "synthesis_started",
            "completed",
            "terminated",
        }
        event_types = [event_type for event_type, _ in events]
        assert set(event_types) <= safe_types

        final_event_type, final_payload = events[-1]
        assert final_event_type in ("completed", "terminated")
        assert final_payload["answer"]
        assert final_payload["sources"], "expected at least one citation in the final payload"
        assert all("cache-invalidation-runbook" in s["source"] for s in final_payload["sources"])
    finally:
        vectorstore.delete_dataset(ns)
        _clear_overrides(_AGENT_OVERRIDE_DEPS)


# ---------------------------------------------------------------------------
# 6. Insufficient evidence: unknown question -> agent terminates correctly
#    rather than hallucinating.
# ---------------------------------------------------------------------------


def test_agent_does_not_hallucinate_a_confident_answer_to_an_unanswerable_question(
    require_postgres, require_ollama, config, tmp_path: Path
):
    """A question about something absent from the entire knowledge base, via POST /agent/query.

    The ingested document is on a completely unrelated topic. A
    well-behaved agent must not fabricate a specific, confident technical
    answer about the fictional entity the question asks about. Combines
    two independent signals (either is individually known to have gaps:
    `_looks_like_refusal` is a literal phrase match with documented
    false-negative cases, and `termination_reason` can legitimately read
    `synthesized` even for a well-phrased refusal) so the assertion isn't
    fragile to either alone.
    """
    ns = f"pytest-e2e-insufficient-{uuid.uuid4()}"
    agentic = _agentic_config(config)
    vectorstore, embedder, _pipeline = _wire_agent_dependencies(agentic)
    ingestion = IngestionPipeline(agentic, vectorstore=vectorstore, embedder=embedder)

    doc = _write_doc(
        tmp_path,
        "vpn-setup-guide.md",
        {},
        "# VPN Setup Guide\n\nInstall the corporate VPN client and authenticate with your "
        "single sign-on credentials before connecting to internal services.",
    )
    ingestion.ingest_file(doc, ns)

    try:
        client = TestClient(app)
        response = client.post(
            "/agent/query",
            json={
                "query": "What is the maximum sustained throughput in petabytes per second "
                "of the Zentrivex flux relay cluster?",
                "filters": {"dataset_id": ns},
            },
        )
        assert response.status_code == 200
        body = response.json()
        answer = body["answer"]
        assert body["termination_reason"] in _VALID_TERMINATIONS
        assert answer
        # The strongest, hardest-to-game signal: the question fishes for one
        # specific fabricatable fact (a throughput number for a fictional
        # cluster). Whatever the model says, it must not confidently invent
        # that number. This holds regardless of how it phrases the
        # refusal, unlike a literal phrase match. `_looks_like_refusal` and
        # `termination_reason` are kept as documented, individually-gappy
        # secondary signals (proven in practice: an earlier run produced
        # "The answer cannot be determined from the provided evidence,"
        # correct behavior that matched neither).
        fabricated_number = re.search(r"\d+(\.\d+)?\s*(petabytes?|pb)\b", answer, re.IGNORECASE)
        looks_like_refusal_or_insufficient = body[
            "termination_reason"
        ] == "insufficient_evidence" or _looks_like_refusal(answer)
        assert (
            not fabricated_number or looks_like_refusal_or_insufficient
        ), f"expected no fabricated throughput figure, got: {answer!r}"
    finally:
        vectorstore.delete_dataset(ns)
        _clear_overrides(_AGENT_OVERRIDE_DEPS)
