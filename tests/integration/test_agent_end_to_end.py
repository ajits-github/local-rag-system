"""Full-stack agentic RAG proof: real Postgres + real Ollama, no fakes, no mocks.

Runs config.agent.enabled=True end to end through run_agent().

Assertions are deliberately robust to real-LLM variability (a local 3B model's JSON-
schema compliance and tool-selection choices aren't perfectly reproducible run to
run): the hard requirements are that the run always completes without raising, always
produces a non-empty final answer, and -- the actual security-critical property --
never leaks another tenant's content into the answer regardless of which tools the
model chose to call. Exact tool sequencing is not asserted here (see the mocked
tests/unit/test_agent_graph_*.py files for deterministic, scripted-LLM proof of the
graph's control flow and bounds).
"""

from __future__ import annotations

import uuid
from pathlib import Path

from rag.agent.graph import run_agent
from rag.agent.state import AgentState
from rag.factory import build_embedder, build_llm, build_vectorstore
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.pipeline import RetrievalPipeline

_VALID_TERMINATIONS = {
    "synthesized",
    "max_steps",
    "max_retrieval_attempts",
    "max_tool_calls",
    "insufficient_evidence",
}


def _agentic_config(config):
    """Return a copy of `config` with the agent enabled, security on, and a more capable model.

    Uses qwen2.5:3b -- matching PROJECT_JOURNAL's finding that it follows
    JSON-schema/refusal instructions substantially more reliably than 1.5b.
    """
    agentic = config.model_copy(deep=True)
    agentic.agent.enabled = True
    agentic.security.authorization.enabled = True
    agentic.generation.model_name = "qwen2.5:3b"
    return agentic


def _write_doc(tmp_path: Path, name: str, frontmatter: dict, body: str) -> Path:
    lines = ["---"]
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


def test_full_multi_hop_question_completes_and_produces_a_grounded_answer(
    require_postgres, require_ollama, config, tmp_path: Path
):
    """The spec's own worked example: an incident points to a service; a runbook holds its fix.

    A separate runbook holds that service's rollback procedure. The run
    must complete and answer, whether or not the model chose to
    decompose/use multiple tool calls to get there.
    """
    ns = f"pytest-agent-e2e-{uuid.uuid4()}"
    agentic = _agentic_config(config)
    ingestion = IngestionPipeline(agentic)

    incident_doc = _write_doc(
        tmp_path,
        "incident-report.md",
        {},
        "# Incident Report\n\n"
        "During the outage, the `ocr-worker` service was rolled back to the previous "
        "release after error rates spiked above threshold.",
    )
    rollback_doc = _write_doc(
        tmp_path,
        "rollback-runbook.md",
        {},
        "# OCR Worker Rollback Runbook\n\n"
        "To roll back ocr-worker, redeploy the last known-good container digest and "
        "verify health checks pass before resuming traffic.",
    )
    result_incident = ingestion.ingest_file(incident_doc, ns)
    result_rollback = ingestion.ingest_file(rollback_doc, ns)

    vectorstore = build_vectorstore(agentic)
    embedder = build_embedder(agentic)
    llm = build_llm(agentic)
    pipeline = RetrievalPipeline(agentic, vectorstore=vectorstore, embedder=embedder, llm=llm)

    try:
        state = AgentState(
            original_query=(
                "Which service was rolled back during the outage, and what is its "
                "rollback procedure?"
            ),
            filters={"dataset_id": ns},
        )
        result = run_agent(
            state,
            pipeline=pipeline,
            vectorstore=vectorstore,
            embedder=embedder,
            llm=llm,
            config=agentic,
        )

        assert result.route in ("classic_rag", "agent")
        assert result.state.termination_reason in _VALID_TERMINATIONS
        assert result.state.final_answer
        assert result.total_ms > 0
        if result.route == "agent":
            # Whatever the model decided to do, it must have stayed within bounds.
            assert result.state.step_count <= agentic.agent.max_agent_steps
            assert result.state.tool_call_count <= agentic.agent.max_tool_calls
            assert result.state.retrieval_attempts <= agentic.agent.max_retrieval_attempts
    finally:
        vectorstore.delete_document(result_incident["document_id"])
        vectorstore.delete_document(result_rollback["document_id"])


def test_agent_path_never_leaks_cross_tenant_content_via_real_model(
    require_postgres, require_ollama, config, tmp_path: Path
):
    """A tenant_alpha caller's agent run never surfaces tenant_beta's secret.

    Holds regardless of what the real model decides to search for or which
    tools it calls.
    """
    ns = f"pytest-agent-e2e-{uuid.uuid4()}"
    agentic = _agentic_config(config)
    ingestion = IngestionPipeline(agentic)

    alpha_doc = _write_doc(
        tmp_path,
        "alpha-rollback.md",
        {"tenant_id": "tenant_alpha", "allowed_roles": ["tenant_alpha_operator"]},
        "The Tenant Alpha emergency rollback code is ALPHA-EMERGENCY-4471.",
    )
    beta_doc = _write_doc(
        tmp_path,
        "beta-rollback.md",
        {"tenant_id": "tenant_beta", "allowed_roles": ["tenant_beta_operator"]},
        "The Tenant Beta emergency rollback code is BETA-EMERGENCY-8823.",
    )
    result_alpha = ingestion.ingest_file(alpha_doc, ns)
    result_beta = ingestion.ingest_file(beta_doc, ns)

    vectorstore = build_vectorstore(agentic)
    embedder = build_embedder(agentic)
    llm = build_llm(agentic)
    pipeline = RetrievalPipeline(agentic, vectorstore=vectorstore, embedder=embedder, llm=llm)

    try:
        auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["tenant_alpha_operator"])
        state = AgentState(
            original_query="What is the emergency rollback code?",
            authorization_context=auth,
            filters={"dataset_id": ns},
        )
        result = run_agent(
            state,
            pipeline=pipeline,
            vectorstore=vectorstore,
            embedder=embedder,
            llm=llm,
            config=agentic,
        )

        assert result.state.final_answer
        assert "BETA-EMERGENCY-8823" not in result.state.final_answer
        for source in result.state.citations:
            assert "beta-rollback" not in source.source
        # Even if the model tried a get_document/get_latest_document call against a
        # guessed Beta path, it would come back empty at the SQL layer -- provable
        # directly (not just via the answer) since sanitized tool evidence is what
        # citations/final_answer are built from.
        for evidence in result.state.retrieved_evidence:
            assert "BETA-EMERGENCY-8823" not in evidence.chunk.content
    finally:
        vectorstore.delete_document(result_alpha["document_id"])
        vectorstore.delete_document(result_beta["document_id"])
