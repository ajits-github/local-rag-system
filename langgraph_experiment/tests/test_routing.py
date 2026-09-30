"""Pure-function tests for classification, extraction, and conditional-edge routing.

No graph invocation, no checkpointer, no I/O -- these are ordinary unit
tests of plain functions.
"""

from __future__ import annotations

from langgraph_experiment.nodes import _extract_case_id, _extract_target_status, classify
from langgraph_experiment.routing import (
    route_after_approval,
    route_after_classify,
    route_after_validate,
)


def test_classify_routes_plain_question_to_read_only():
    result = classify({"original_query": "What is the password rotation policy?"})
    assert result["route"] == "read_only"
    assert result["is_case_mutation"] is False


def test_classify_routes_case_status_question_to_case_read():
    result = classify({"original_query": "What is the status of CASE-2001?"})
    assert result["route"] == "case_read"
    assert result["case_id"] == "CASE-2001"


def test_classify_routes_close_request_to_case_write():
    result = classify({"original_query": "Please close CASE-1001"})
    assert result["route"] == "case_write"
    assert result["is_case_mutation"] is True
    assert result["case_id"] == "CASE-1001"
    assert result["requested_new_status"] == "closed"


def test_classify_routes_resolve_request_to_case_write():
    result = classify({"original_query": "Can you resolve case CASE-2002 please"})
    assert result["route"] == "case_write"
    assert result["requested_new_status"] == "resolved"


def test_classify_directive_phrasing_extracts_status_value():
    result = classify({"original_query": "Set CASE-1001 to in_progress"})
    assert result["route"] == "case_write"
    assert result["requested_new_status"] == "in_progress"


def test_classify_case_mutation_with_status_but_no_case_id():
    # "update case to resolved" clearly names a mutation (directive verb +
    # explicit status value) but no CASE-<digits> token -- classify still
    # routes to case_write (a mutation attempt was clearly named) with a
    # resolvable target status but case_id=None; validate_write_request is
    # what turns the missing case_id into a graceful "invalid_request"
    # refusal rather than pausing for approval on an unknown case.
    result = classify({"original_query": "update case to resolved"})
    assert result["route"] == "case_write"
    assert result["case_id"] is None
    assert result["requested_new_status"] == "resolved"


def test_extract_case_id_case_insensitive():
    assert _extract_case_id("tell me about case-1001") == "CASE-1001"
    assert _extract_case_id("no case id here") is None


def test_extract_target_status_variants():
    assert _extract_target_status("close CASE-1001") == "closed"
    assert _extract_target_status("reopen CASE-1003") == "open"
    assert _extract_target_status("mark CASE-1001 as resolved") == "resolved"
    assert _extract_target_status("change CASE-1001 to in-progress") == "in_progress"
    assert _extract_target_status("what is happening with CASE-1001") is None


def test_route_after_classify_defaults_to_read_only_when_unset():
    assert route_after_classify({}) == "read_only"


def test_route_after_validate_branches_on_pending_action():
    assert route_after_validate({"pending_action": {"case_id": "CASE-1001"}}) == "await_approval"
    assert route_after_validate({}) == "reject_invalid"


def test_route_after_approval_requires_approved_state():
    assert route_after_approval({"pending_action": {"approval_state": "approved"}}) == "execute"
    assert route_after_approval({"pending_action": {"approval_state": "rejected"}}) == "rejected"
    assert (
        route_after_approval({"pending_action": {"approval_state": "denied_insufficient_role"}})
        == "rejected"
    )
    assert route_after_approval({}) == "rejected"
