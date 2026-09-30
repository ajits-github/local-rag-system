"""The case-read branch: classify -> select_case_tool -> execute_case_read_tool -> synthesize.

Uses the real `rag.mcp.business.store` synthetic case data (CASE-1001,
tenant_alpha, roles tenant_alpha_operator/tenant_alpha_admin) -- see
`conftest.py`'s `_restore_synthetic_cases` autouse fixture for why this
is safe against a read-only test.
"""

from __future__ import annotations


def test_authorized_full_case_read(graph, thread_config):
    result = graph.invoke(
        {
            "original_query": "Give me details on CASE-1001",
            "tenant_id": "tenant_alpha",
            "roles": ["tenant_alpha_operator"],
        },
        thread_config,
    )
    assert result["route"] == "case_read"
    assert result["case_tool_name"] == "get_customer_case"
    assert result["termination_reason"] == "synthesized"
    assert "CASE-1001" in result["final_answer"]
    assert "SSO migration" in result["final_answer"]


def test_authorized_status_only_read(graph, thread_config):
    result = graph.invoke(
        {
            "original_query": "What is the status of CASE-1001?",
            "tenant_id": "tenant_alpha",
            "roles": ["tenant_alpha_operator"],
        },
        thread_config,
    )
    assert result["case_tool_name"] == "get_case_status"
    assert "in_progress" in result["final_answer"]


def test_unauthorized_cross_tenant_read_is_denied_not_found(graph, thread_config):
    """A caller from the wrong tenant, no support role, reads the same as a nonexistent case."""
    result = graph.invoke(
        {
            "original_query": "Give me details on CASE-1001",
            "tenant_id": "tenant_beta",
            "roles": ["tenant_beta_operator"],
        },
        thread_config,
    )
    assert result["termination_reason"] == "case_not_found_or_denied"
    assert "not authorized" in result["final_answer"] or "could not find" in result["final_answer"]


def test_nonexistent_case_reads_as_not_found(graph, thread_config):
    result = graph.invoke(
        {
            "original_query": "Give me details on CASE-9999",
            "tenant_id": "tenant_alpha",
            "roles": [],
        },
        thread_config,
    )
    assert result["case_found"] is False
    assert result["termination_reason"] == "case_not_found_or_denied"
