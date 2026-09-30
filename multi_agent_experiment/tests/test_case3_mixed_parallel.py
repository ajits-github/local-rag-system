"""CASE 3 (required scenario): mixed question -> Knowledge + Business Agents in parallel.

Also proves the parallel dispatch is genuine (both specialists' tool calls
are recorded, both fired from the same coordinator round) rather than one
silently short-circuiting the other.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from multi_agent_experiment.graph import build_graph


def test_mixed_question_runs_both_specialists_and_grounds_the_answer_in_both(
    deps_with_retrieval, thread_config
):
    """A mixed query dispatches both specialists and grounds the answer in both sources."""
    graph = build_graph(deps_with_retrieval, InMemorySaver())
    result = graph.invoke(
        {
            "original_query": (
                "According to the support policy, what should happen next for CASE-1001?"
            ),
            "tenant_id": "tenant_alpha",
            "roles": ["tenant_alpha_operator"],
        },
        thread_config,
    )
    assert result["selected_specialists"] == ["knowledge", "business"]
    assert result["needs_knowledge"] is True
    assert result["needs_business"] is True
    assert result["is_case_mutation"] is False

    # Both specialists actually ran and contributed a tool call.
    tool_specialists = {call["specialist"] for call in result["tool_call_log"]}
    assert tool_specialists == {"knowledge", "business"}

    # Both evidence sources reached final synthesis.
    assert result["knowledge_status"] == "ok"
    assert result["business_status"] == "ok"
    assert result["case_result"]["case_id"] == "CASE-1001"
    assert len(result["citations"]) == 2
    assert {c["source"] for c in result["citations"]} == {"docs/api-policy.md", "case:CASE-1001"}

    assert result["termination_reason"] == "synthesized"
    # The fake LLM was called once, over context built from BOTH sources --
    # assert the rendered prompt (captured by FakePromptTemplate) actually
    # contains both, proving this is one real merged-context answer, not
    # two independent answers concatenated.
    _, user_prompt = deps_with_retrieval.llm.calls[-1]
    assert "docs/api-policy.md" in user_prompt
    assert "CASE-1001" in user_prompt


def test_mixed_route_dispatches_both_branches_in_the_same_superstep(
    deps_with_retrieval, thread_config
):
    """`stream_mode="debug"` shows both specialist tasks sharing one step, not sequential steps.

    `stream_mode="updates"` was tried first and rejected: LangGraph emits
    one `updates` chunk per *node completion*, not one per superstep, so
    two genuinely parallel branches still show up as separate chunks --
    confirmed empirically before writing this assertion (see
    `tests/test_probe_langgraph_primitives.py` for the same
    verify-against-the-real-library discipline). `"debug"` mode's `task`
    events carry an explicit `step` number, which *is* the right signal:
    both `knowledge_agent` and `business_select_tool` must be dispatched
    as `task` events under the identical step number.
    """
    graph = build_graph(deps_with_retrieval, InMemorySaver())
    task_steps: dict[str, int] = {}
    for chunk in graph.stream(
        {
            "original_query": "According to policy, what about CASE-1001?",
            "tenant_id": "tenant_alpha",
            "roles": ["tenant_alpha_operator"],
        },
        thread_config,
        stream_mode="debug",
    ):
        if chunk.get("type") == "task":
            task_steps[chunk["payload"]["name"]] = chunk["step"]

    assert "knowledge_agent" in task_steps
    assert "business_select_tool" in task_steps
    assert task_steps["knowledge_agent"] == task_steps["business_select_tool"]
