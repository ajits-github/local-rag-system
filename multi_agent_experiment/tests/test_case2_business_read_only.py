"""CASE 2 (required scenario): pure business question -> coordinator -> business_agent -> answer.

The Knowledge Agent must not run at all.
"""

from __future__ import annotations


def test_pure_business_question_never_invokes_knowledge_agent(graph, thread_config):
    """Business-only routing must never touch any knowledge-branch field."""
    result = graph.invoke(
        {
            "original_query": "What is the status of CASE-1001?",
            "tenant_id": "tenant_alpha",
            "roles": ["tenant_alpha_operator"],
        },
        thread_config,
    )
    assert result["needs_knowledge"] is False
    assert result["needs_business"] is True
    assert result["selected_specialists"] == ["business"]
    assert result["case_tool_name"] == "get_case_status"
    assert result["business_status"] == "ok"
    assert result["termination_reason"] == "synthesized"
    assert "in_progress" in result["final_answer"]
    # The knowledge branch never ran.
    assert result.get("knowledge_evidence") in (None, [])
    assert result.get("knowledge_status") in (None, "not_run")
    assert result["citations"] == [
        {"chunk_id": None, "case_id": "CASE-1001", "source": "case:CASE-1001", "score": None}
    ]
    assert "__interrupt__" not in result


def test_business_only_question_with_no_evidence_never_fabricates(graph, thread_config):
    """CASE 5 companion: an authorization denial on a business-only route must not fabricate."""
    result = graph.invoke(
        {
            "original_query": "What is the status of CASE-1002?",
            "tenant_id": "tenant_alpha",
            # CASE-1002 is admin-only within tenant_alpha; operator lacks access.
            "roles": ["tenant_alpha_operator"],
        },
        thread_config,
    )
    assert result["business_status"] == "not_found_or_denied"
    # The more specific reason business_agent.evaluate_read already set is
    # preserved rather than overwritten by a generic "insufficient_evidence".
    assert result["termination_reason"] == "case_not_found_or_denied"
    assert "don't have enough" in result["final_answer"]
    assert result["citations"] == []
    assert any("business specialist returned" in note for note in result["critic_notes"])
