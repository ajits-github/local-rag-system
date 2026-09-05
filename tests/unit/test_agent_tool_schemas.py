from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag.agent.tool_schemas import (
    TOOL_ARG_MODELS,
    GetCaseStatusArgs,
    GetCustomerCaseArgs,
    GetDocumentArgs,
    GetLatestDocumentArgs,
    GetRelatedContextArgs,
    SearchKnowledgeBaseArgs,
    UpdateCaseStatusArgs,
)


def test_search_knowledge_base_args_rejects_smuggled_privileged_field():
    """extra='forbid' rejects an LLM-supplied tenant_id/roles/auth-shaped key.

    This is the structural (not just prompted) guarantee that the LLM
    cannot supply or override authorization. see docs/architecture.md's
    "Agentic RAG" section.
    """
    with pytest.raises(ValidationError):
        SearchKnowledgeBaseArgs.model_validate(
            {"query": "hello", "tenant_id": "tenant_beta", "roles": ["security_admin"]}
        )


def test_get_document_args_rejects_smuggled_auth_field():
    """extra='forbid' rejects an LLM-supplied auth-shaped key on GetDocumentArgs."""
    with pytest.raises(ValidationError):
        GetDocumentArgs.model_validate({"source": "a.md", "auth": {"tenant_id": "tenant_beta"}})


def test_get_latest_document_args_rejects_extra_field():
    """extra='forbid' rejects an undeclared field (as_of) on GetLatestDocumentArgs."""
    with pytest.raises(ValidationError):
        GetLatestDocumentArgs.model_validate({"source": "a.md", "as_of": "2026-01-01"})


def test_get_related_context_args_rejects_extra_field():
    """extra='forbid' rejects an LLM-supplied roles-shaped key on GetRelatedContextArgs."""
    with pytest.raises(ValidationError):
        GetRelatedContextArgs.model_validate({"chunk_id": "c1", "roles": ["security_admin"]})


def test_search_knowledge_base_top_k_defaults_and_bounds():
    """top_k has a hard Field(ge=1, le=20) range. the LLM can request less, never more."""
    args = SearchKnowledgeBaseArgs.model_validate({"query": "hello"})
    assert args.top_k == 5

    with pytest.raises(ValidationError):
        SearchKnowledgeBaseArgs.model_validate({"query": "hello", "top_k": 21})

    with pytest.raises(ValidationError):
        SearchKnowledgeBaseArgs.model_validate({"query": "hello", "top_k": 0})

    args = SearchKnowledgeBaseArgs.model_validate({"query": "hello", "top_k": 20})
    assert args.top_k == 20


def test_get_document_args_expose_no_chunk_count_field():
    """get_document/get_latest_document schemas expose no chunk-limit field at all.

    The chunk-count bound is entirely server-controlled (rag.config.AgentConfig);
    nothing in these schemas lets an LLM request "all chunks" or any
    specific chunk count.
    """
    assert "limit" not in GetDocumentArgs.model_fields
    assert "max_chunks" not in GetDocumentArgs.model_fields
    assert "limit" not in GetLatestDocumentArgs.model_fields
    assert "max_chunks" not in GetLatestDocumentArgs.model_fields


def test_tool_arg_models_cover_every_tool_name():
    """TOOL_ARG_MODELS has exactly one entry per tool the agent can dispatch.

    Seven tools: the four local RAG tools plus the three remote MCP
    business tools (get_customer_case/get_case_status/update_case_status).
    """
    assert set(TOOL_ARG_MODELS) == {
        "search_knowledge_base",
        "get_document",
        "get_latest_document",
        "get_related_context",
        "get_customer_case",
        "get_case_status",
        "update_case_status",
    }


def test_get_customer_case_args_rejects_smuggled_auth_field():
    """extra='forbid' rejects an LLM-supplied auth-shaped key on GetCustomerCaseArgs."""
    with pytest.raises(ValidationError):
        GetCustomerCaseArgs.model_validate({"case_id": "CASE-1001", "tenant_id": "tenant_evil"})


def test_get_case_status_args_rejects_smuggled_auth_field():
    """extra='forbid' rejects an LLM-supplied roles-shaped key on GetCaseStatusArgs."""
    with pytest.raises(ValidationError):
        GetCaseStatusArgs.model_validate({"case_id": "CASE-1001", "roles": ["security_admin"]})


def test_update_case_status_args_accepts_only_case_id_and_new_status():
    """UpdateCaseStatusArgs has no approval field; extra='forbid' rejects one anyway."""
    args = UpdateCaseStatusArgs.model_validate({"case_id": "CASE-1001", "new_status": "closed"})
    assert args.case_id == "CASE-1001"
    assert args.new_status == "closed"

    with pytest.raises(ValidationError):
        UpdateCaseStatusArgs.model_validate(
            {"case_id": "CASE-1001", "new_status": "closed", "approved": True}
        )
    with pytest.raises(ValidationError):
        UpdateCaseStatusArgs.model_validate({"case_id": "CASE-1001", "new_status": "not-a-status"})
