"""CASE 5 (required scenario): a specialist fails, and the Coordinator never fabricates.

Covers: a Knowledge Agent exception (simulated timeout), a Business Agent
authorization denial (covered more fully in `test_case2_business_read_
only.py`'s second test), and the bounded knowledge-retry loop actually
recovering when the second attempt succeeds.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from langgraph_experiment.wiring import GraphDeps
from multi_agent_experiment.graph import build_graph
from multi_agent_experiment.limits import MAX_KNOWLEDGE_RETRIES
from multi_agent_experiment.tests.conftest import (
    FakeLLM,
    FakePipeline,
    FakePromptTemplate,
    make_search_result,
)
from rag.config import load_config


def test_knowledge_agent_exception_is_contained_and_never_crashes_the_run(thread_config):
    """A raised exception from the Knowledge Agent's tool must never crash the run."""
    config = load_config()
    deps = GraphDeps(
        config=config,
        pipeline=FakePipeline(raises=TimeoutError),
        llm=FakeLLM("should never be reached"),
        rag_prompt_template=FakePromptTemplate(),
    )
    graph = build_graph(deps, InMemorySaver())
    result = graph.invoke(
        {"original_query": "What is the current API timeout policy?"}, thread_config
    )
    assert result["knowledge_status"] == "failed"
    assert result["knowledge_error"] == "TimeoutError"
    # No answer was fabricated from the failed call.
    assert "don't have enough" in result["final_answer"]
    assert result["citations"] == []
    assert deps.llm.calls == []  # final_synthesis never even reached the LLM


def test_knowledge_agent_retries_once_then_gives_up_within_the_bound(thread_config):
    """A persistently failing Knowledge Agent retries exactly once, then stops."""
    config = load_config()
    pipeline = FakePipeline(raises=TimeoutError)
    deps = GraphDeps(
        config=config,
        pipeline=pipeline,
        llm=FakeLLM("should never be reached"),
        rag_prompt_template=FakePromptTemplate(),
    )
    graph = build_graph(deps, InMemorySaver())
    result = graph.invoke(
        {"original_query": "What is the current API timeout policy?"}, thread_config
    )
    # Exactly one retry: the initial attempt plus MAX_KNOWLEDGE_RETRIES.
    assert len(pipeline.calls) == 1 + MAX_KNOWLEDGE_RETRIES
    assert result["retry_count"] == MAX_KNOWLEDGE_RETRIES
    assert result["termination_reason"] == "max_rounds"
    assert any("retry budget was exhausted" in note for note in result["critic_notes"])


def test_knowledge_agent_retry_recovers_when_the_second_attempt_succeeds(thread_config):
    """A bounded retry recovers when the second, widened attempt finds evidence."""
    config = load_config()
    good_result = make_search_result(
        "c9", "docs/sla.md", "The support SLA response window is 4 hours."
    )
    # First call (default top_k) finds nothing; the critic-widened retry finds the chunk.
    pipeline = FakePipeline(results_by_call=[[], [good_result]])
    deps = GraphDeps(
        config=config,
        pipeline=pipeline,
        llm=FakeLLM("The SLA response window is 4 hours, per docs/sla.md."),
        rag_prompt_template=FakePromptTemplate(),
    )
    graph = build_graph(deps, InMemorySaver())
    result = graph.invoke(
        {"original_query": "What is the support SLA response window?"}, thread_config
    )

    assert len(pipeline.calls) == 2
    assert pipeline.calls[0]["candidate_k"] < pipeline.calls[1]["candidate_k"]
    assert result["retry_count"] == 1
    assert result["knowledge_status"] == "ok"
    assert result["termination_reason"] == "synthesized"
    assert "4 hours" in result["final_answer"]
    assert any("requesting one bounded retry" in note for note in result["critic_notes"])
