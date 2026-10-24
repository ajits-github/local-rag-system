"""The normal, non-agentic read-only RAG path: classify -> retrieve -> synthesize_rag -> END."""

from __future__ import annotations


def test_read_only_query_runs_straight_through_to_synthesized_answer(
    graph_with_retrieval, thread_config
):
    result = graph_with_retrieval.invoke(
        {"original_query": "What is the password rotation policy?"}, thread_config
    )
    assert result["route"] == "read_only"
    assert result["termination_reason"] == "synthesized"
    assert "90 days" in result["final_answer"]
    assert result["citations"][0]["source"] == "docs/policy.md"
    # No interrupt on this path.
    assert "__interrupt__" not in result


def test_read_only_query_with_no_evidence_reports_insufficient(deps_with_retrieval, thread_config):
    from langgraph.checkpoint.memory import InMemorySaver

    from langgraph_experiment.graph import build_graph

    deps_with_retrieval.pipeline._results = []  # noqa: SLF001 -- test-only introspection of the fake
    graph = build_graph(deps_with_retrieval, InMemorySaver())
    result = graph.invoke({"original_query": "anything"}, thread_config)
    assert result["termination_reason"] == "insufficient_evidence"
    assert result["citations"] == []


def test_read_only_query_without_pipeline_reports_unavailable(deps, graph, thread_config):
    # `deps` (the case-store-only fixture) has pipeline=None -- the graph
    # must still terminate gracefully rather than crash.
    result = graph.invoke(
        {"original_query": "What is the password rotation policy?"}, thread_config
    )
    assert result["termination_reason"] == "retrieval_unavailable"
