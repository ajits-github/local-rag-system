"""Pure, deterministic tests for `orchestrator.coordinator` and `routing.py` -- no graph run."""

from __future__ import annotations

from langgraph.types import Send

from multi_agent_experiment import orchestrator, routing
from multi_agent_experiment.limits import DEFAULT_KNOWLEDGE_TOP_K, RETRY_KNOWLEDGE_TOP_K


def test_coordinator_routes_pure_knowledge_question_to_knowledge_only():
    """A pure knowledge question selects only the Knowledge Agent."""
    result = orchestrator.coordinator({"original_query": "What is the current API timeout policy?"})
    assert result["needs_knowledge"] is True
    assert result["needs_business"] is False
    assert result["is_case_mutation"] is False
    assert result["selected_specialists"] == ["knowledge"]
    assert result["knowledge_top_k"] == DEFAULT_KNOWLEDGE_TOP_K


def test_coordinator_routes_pure_business_question_to_business_only():
    """A pure business question selects only the Business Agent."""
    result = orchestrator.coordinator({"original_query": "What is the status of CASE-1001?"})
    assert result["needs_knowledge"] is False
    assert result["needs_business"] is True
    assert result["is_case_mutation"] is False
    assert result["case_id"] == "CASE-1001"
    assert result["selected_specialists"] == ["business"]


def test_coordinator_routes_mixed_question_to_both_specialists():
    """A mixed question selects both specialists."""
    result = orchestrator.coordinator(
        {
            "original_query": (
                "According to the support policy, what should happen next for CASE-1001?"
            )
        }
    )
    assert result["needs_knowledge"] is True
    assert result["needs_business"] is True
    assert result["is_case_mutation"] is False
    assert result["selected_specialists"] == ["knowledge", "business"]


def test_coordinator_routes_mutation_request_as_case_mutation():
    """A mutation-shaped query is classified as a case mutation."""
    result = orchestrator.coordinator({"original_query": "Please close CASE-2001"})
    assert result["is_case_mutation"] is True
    assert result["case_id"] == "CASE-2001"
    assert result["requested_new_status"] == "closed"


def test_coordinator_retry_round_widens_knowledge_top_k_and_disables_business():
    """A retry round widens knowledge_top_k and disables the business branch."""
    result = orchestrator.coordinator(
        {"original_query": "irrelevant on a retry round", "coordinator_round": 1}
    )
    assert result["coordinator_round"] == 2
    assert result["knowledge_top_k"] == RETRY_KNOWLEDGE_TOP_K
    assert result["needs_business"] is False
    # A retry round must not re-derive routing/case fields.
    assert "selected_specialists" not in result
    assert "is_case_mutation" not in result


def test_route_after_coordinator_mutation_returns_bare_string():
    """A mutation route returns the bare destination string, not a Send."""
    state = {"is_case_mutation": True}
    assert routing.route_after_coordinator(state) == "business_validate_write"


def test_route_after_coordinator_knowledge_only_returns_one_send():
    """A knowledge-only route returns exactly one Send targeting knowledge_agent."""
    state = {"is_case_mutation": False, "needs_knowledge": True, "needs_business": False}
    targets = routing.route_after_coordinator(state)
    assert len(targets) == 1
    assert isinstance(targets[0], Send)
    assert targets[0].node == "knowledge_agent"


def test_route_after_coordinator_mixed_returns_two_parallel_sends():
    """A mixed route returns two Sends, one per specialist."""
    state = {"is_case_mutation": False, "needs_knowledge": True, "needs_business": True}
    targets = routing.route_after_coordinator(state)
    assert {t.node for t in targets} == {"knowledge_agent", "business_select_tool"}


def test_route_after_critic_reads_the_flag():
    """The critic's retry routing reads `critic_wants_retry`, defaulting to proceed."""
    assert routing.route_after_critic({"critic_wants_retry": True}) == "retry"
    assert routing.route_after_critic({"critic_wants_retry": False}) == "proceed"
    assert routing.route_after_critic({}) == "proceed"
