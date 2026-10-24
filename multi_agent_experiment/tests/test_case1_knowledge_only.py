"""CASE 1 (required scenario): pure knowledge question -> coordinator -> knowledge_agent -> answer.

The Business Agent must not run at all.
"""

from __future__ import annotations


def test_pure_knowledge_question_never_invokes_business_agent(graph_with_retrieval, thread_config):
    """Knowledge-only routing must never touch any business-branch field."""
    result = graph_with_retrieval.invoke(
        {"original_query": "What is the current API timeout policy?"},
        thread_config,
    )
    assert result["needs_knowledge"] is True
    assert result["needs_business"] is False
    assert result["selected_specialists"] == ["knowledge"]
    assert result["termination_reason"] == "synthesized"
    assert "30 seconds" in result["final_answer"]
    # The business branch never ran: none of its fields were ever set.
    assert result.get("case_id") is None
    assert result.get("case_result") is None
    assert result.get("business_status") in (None, "not_run")
    assert result["citations"][0]["source"] == "docs/api-policy.md"
    assert result["llm_call_count"] == 1
    assert "__interrupt__" not in result
