"""Tool-argument recording and diagnostics in the agent eval."""

from __future__ import annotations

import json

from rag.agent.state import ToolCallRecord
from rag.eval.gold_schema import GoldExample
from rag.eval.run_agent_eval import (
    _error_category,
    _safe_tool_args,
    _tool_argument_diagnostics,
    _tool_call_record,
)


def _call(tool_name, args, **extra):
    return {
        "tool_name": tool_name,
        "tool_args": args,
        "success": True,
        "result_count": 1,
        "error_category": None,
        **extra,
    }


def _row(calls, expected, **extra):
    return {
        "tool_call_records": calls,
        "tool_calls": [c["tool_name"] for c in calls],
        "expected_tool_args": expected,
        **extra,
    }


def test_safe_tool_args_keeps_only_declared_fields():
    """A forged identity or credential key can never reach a report."""
    forged = {
        "case_id": "CASE-1",
        "tenant_id": "other",
        "roles": ["admin"],
        "authorization": "Bearer abc",
        "token": "abc",
        "auth": {"tenant_id": "other"},
    }
    assert _safe_tool_args("get_case_status", forged) == {"case_id": "CASE-1"}


def test_tool_call_record_is_json_safe_and_carries_no_identity_or_token():
    """Check: tool call record is json safe and carries no identity or token."""
    record = ToolCallRecord(
        tool_name="update_case_status",
        args={"case_id": "CASE-1", "new_status": "closed", "case_approvals": ["x"], "jwt": "abc"},
        result_count=1,
    )
    rendered = _tool_call_record(record, set())

    assert rendered["tool_args"] == {"case_id": "CASE-1", "new_status": "closed"}
    serialized = json.dumps(rendered)
    for secret in ("jwt", "case_approvals", "abc"):
        assert secret not in serialized


def test_error_category_never_passes_free_form_text_through():
    """Check: error category never passes free form text through."""
    assert _error_category(None) is None
    assert _error_category("invalid_arguments") == "invalid_arguments"
    assert _error_category("graph_disabled") == "graph_disabled"
    assert _error_category("boom: token=abc") == "execution_error"


def test_related_context_records_whether_the_id_was_shown_in_evidence():
    """Check: related context records whether the id was shown in evidence."""
    record = ToolCallRecord(tool_name="get_related_context", args={"chunk_id": "real_1"})
    assert _tool_call_record(record, {"real_1"})["chunk_id_from_evidence"] is True
    assert _tool_call_record(record, {"other"})["chunk_id_from_evidence"] is False


def test_diagnostics_separate_wrong_argument_from_wrong_tool():
    """Check: diagnostics separate wrong argument from wrong tool."""
    records = [
        _row(
            [_call("get_case_status", {"case_id": "CASE-1"})],
            {"get_case_status": {"case_id": "CASE-1"}},
        ),
        _row(
            [_call("get_case_status", {"case_id": "CASE-9"})],
            {"get_case_status": {"case_id": "CASE-1"}},
        ),
        _row(
            [_call("search_knowledge_base", {"query": "CASE-1"})],
            {"get_case_status": {"case_id": "CASE-1"}},
        ),
    ]
    result = _tool_argument_diagnostics(records)["tool_argument_diagnostics"]

    assert result["correct_tool_correct_args"] == 1
    assert result["correct_tool_wrong_args"] == 1
    assert result["expected_tool_not_called"] == 1


def test_diagnostics_match_source_by_suffix_and_accept_any_listed_version():
    """Check: diagnostics match source by suffix and accept any listed version."""
    expected = {"get_latest_document": {"source": ["kb/p-v1.md", "kb/p-v2.md"]}}
    for source in ("data/kb/p-v2.md", "data\\kb\\p-v1.md"):
        records = [_row([_call("get_latest_document", {"source": source})], expected)]
        diagnostics = _tool_argument_diagnostics(records)["tool_argument_diagnostics"]
        assert diagnostics["correct_tool_correct_args"] == 1

    records = [_row([_call("get_latest_document", {"source": "data/kb/other.md"})], expected)]
    diagnostics = _tool_argument_diagnostics(records)["tool_argument_diagnostics"]
    assert diagnostics["correct_tool_wrong_args"] == 1


def test_diagnostics_compare_status_exactly():
    """Check: diagnostics compare status exactly."""
    expected = {"update_case_status": {"case_id": "CASE-1", "new_status": "closed"}}
    wrong = _call("update_case_status", {"case_id": "CASE-1", "new_status": "resolved"})
    diagnostics = _tool_argument_diagnostics([_row([wrong], expected)])
    assert diagnostics["tool_argument_diagnostics"]["correct_tool_wrong_args"] == 1


def test_diagnostics_count_validation_errors_parse_failures_and_case_row_searches():
    """Check: diagnostics count validation errors parse failures and case row searches."""
    invalid = _call("get_document", {}) | {"error_category": "invalid_arguments"}
    records = [
        _row([invalid], {}, tool_selection_parse_failures=2),
        _row(
            [_call("search_knowledge_base", {"query": "x"})],
            {},
            requires_mcp_business_tool=True,
        ),
    ]
    result = _tool_argument_diagnostics(records)["tool_argument_diagnostics"]

    assert result["argument_validation_errors"] == 1
    assert result["tool_selection_parse_failures"] == 2
    assert result["kb_search_on_case_rows"] == 1


def test_gold_schema_accepts_expected_tool_args():
    """Check: gold schema accepts expected tool args."""
    example = GoldExample(
        question="q",
        expected_answer="a",
        expected_tool_args={"get_case_status": {"case_id": "CASE-1"}},
    )
    assert example.expected_tool_args["get_case_status"]["case_id"] == "CASE-1"
