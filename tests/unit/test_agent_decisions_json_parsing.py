from __future__ import annotations

import pytest

from rag.agent.decisions import ClassifyDecision, parse_llm_json, run_decision
from rag.prompts.loader import PromptTemplate


class FakeLLM:
    """LLM double returning a fixed sequence of responses, one per call."""

    def __init__(self, responses: list[str]) -> None:
        """Store the queued responses and start with no recorded calls."""
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        """Record the call and return the next queued response."""
        self.calls.append((system, user))
        return self._responses.pop(0)

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


def _template() -> PromptTemplate:
    """Build a minimal single-variable prompt template for decision tests."""
    return PromptTemplate(
        prompt_id="test_classify",
        version="v-test",
        description="test",
        system_template="classify",
        user_template="Question: {query}",
        required_variables=["query"],
        created_at="2026-08-15",
    )


def test_parse_llm_json_valid_object():
    """A well-formed JSON object parses and validates directly."""
    decision = parse_llm_json('{"query_type": "simple", "reasoning": "ok"}', ClassifyDecision)
    assert decision.query_type == "simple"


def test_parse_llm_json_tolerates_surrounding_prose():
    """A JSON object embedded in surrounding prose is still extracted and parsed."""
    raw = 'Sure, here is the answer:\n{"query_type": "complex"}\nHope that helps!'
    decision = parse_llm_json(raw, ClassifyDecision)
    assert decision.query_type == "complex"


def test_parse_llm_json_raises_on_malformed_json():
    """Non-JSON text raises a ValueError naming the failure as a JSON parse error."""
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_llm_json("not json at all", ClassifyDecision)


def test_parse_llm_json_raises_on_schema_mismatch():
    """Valid JSON that doesn't match the target schema raises a ValueError."""
    with pytest.raises(ValueError, match="does not match"):
        parse_llm_json('{"query_type": "not_a_valid_choice"}', ClassifyDecision)


def test_run_decision_succeeds_on_first_valid_response():
    """A valid first response is parsed with exactly one LLM call, no retry."""
    llm = FakeLLM(['{"query_type": "simple"}'])
    decision = run_decision(llm, _template(), ClassifyDecision, max_retries=1, query="hi")
    assert decision is not None
    assert decision.query_type == "simple"
    assert len(llm.calls) == 1


def test_run_decision_retries_once_then_succeeds():
    """A malformed first response gets one bounded reparse nudge before succeeding."""
    llm = FakeLLM(["not json", '{"query_type": "complex"}'])
    decision = run_decision(llm, _template(), ClassifyDecision, max_retries=1, query="hi")
    assert decision is not None
    assert decision.query_type == "complex"
    assert len(llm.calls) == 2
    # The retry nudge is appended to the user turn, not silently dropped.
    assert "not valid JSON" in llm.calls[1][1]


def test_run_decision_returns_none_after_exhausting_retries():
    """Every attempt failing returns None -- callers must supply a safe default, never crash."""
    llm = FakeLLM(["nope", "still nope"])
    decision = run_decision(llm, _template(), ClassifyDecision, max_retries=1, query="hi")
    assert decision is None
    assert len(llm.calls) == 2


def test_run_decision_zero_retries_makes_exactly_one_call():
    """max_retries=0 makes exactly one LLM call, no reparse nudge, on a parse failure."""
    llm = FakeLLM(["not json"])
    decision = run_decision(llm, _template(), ClassifyDecision, max_retries=0, query="hi")
    assert decision is None
    assert len(llm.calls) == 1
