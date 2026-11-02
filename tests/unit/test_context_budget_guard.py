"""An infeasible prompt never reaches a provider, on any generation path (CTX-015)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rag.agent.graph import _synthesize
from rag.agent.state import AgentState
from rag.config import load_config
from rag.context.engine import ContextBudgetExceededError, ContextEngine
from rag.graph.pipeline import GraphRAGPipeline
from rag.retrieval.pipeline import RetrievalPipeline


def _config(window=1024, max_tokens=512):
    config = load_config().model_copy(deep=True)
    config.context.enabled = True
    config.context.model_limits.overrides = {}
    config.context.model_limits.default_context_window_tokens = window
    config.generation.max_tokens = max_tokens
    return config


def _pipeline(config) -> RetrievalPipeline:
    pipeline = object.__new__(RetrievalPipeline)
    pipeline._context_engine = ContextEngine(config)
    pipeline._retrieve_timed = Mock(return_value=([], {}))
    pipeline._prompt_template = SimpleNamespace(
        system_template="system", render=lambda context, query: ("system", query + context)
    )
    pipeline._llm = Mock()
    pipeline._llm.generate.return_value = "answer"
    return pipeline


def test_answer_never_calls_the_provider_for_an_infeasible_prompt():
    """Empty evidence does not make the mandatory prompt fit."""
    pipeline = _pipeline(_config())

    with pytest.raises(ContextBudgetExceededError) as exc:
        pipeline.answer("q " * 6000)

    assert exc.value.overflow_tokens > 0
    pipeline._llm.generate.assert_not_called()


def test_answer_generates_when_the_prompt_fits():
    """The control: a normal request still reaches the provider."""
    pipeline = _pipeline(_config(window=32768))

    assert pipeline.answer("short question")["answer"] == "answer"
    pipeline._llm.generate.assert_called_once()


def test_graph_generation_path_is_guarded():
    """`GraphRAGPipeline` renders through the same guard."""
    config = _config()
    retrieval = _pipeline(config)
    graphrag = object.__new__(GraphRAGPipeline)
    graphrag._retrieval_pipeline = retrieval

    with pytest.raises(ContextBudgetExceededError):
        graphrag._generate_from_results("q " * 6000, [], 0.0, 0.0)

    retrieval._llm.generate.assert_not_called()


def test_agent_synthesis_path_is_guarded():
    """The agent's synthesis node renders through the same guard."""
    pipeline = _pipeline(_config())
    llm = Mock()
    template = SimpleNamespace(system_template="system", render=Mock())

    with pytest.raises(ContextBudgetExceededError):
        _synthesize(AgentState(original_query="q " * 6000), llm, template, pipeline)

    llm.generate.assert_not_called()


def test_generation_limit_above_the_reservation_counts_against_the_window():
    """The provider may generate `max_tokens`, whatever the reservation says."""
    config = _config(window=1024, max_tokens=4096)
    config.context.reserved_output_tokens = 100
    engine = ContextEngine(config)

    packed = engine.build(query="q", system_prompt_text="", search_results=[])

    assert packed.budget.infeasible is False
    assert engine.prompt_overflow_tokens(packed) > 0
    with pytest.raises(ContextBudgetExceededError):
        engine.require_prompt_fits(packed)


def test_api_returns_413_without_echoing_the_prompt():
    """The error reaches a caller as a client error, not a 500 with internals."""
    from rag.api.main import _handle_context_budget_exceeded

    response = _handle_context_budget_exceeded(
        SimpleNamespace(url=SimpleNamespace(path="/query")), ContextBudgetExceededError(10)
    )

    assert response.status_code == 413
    assert b"10" not in response.body
