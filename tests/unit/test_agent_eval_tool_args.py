"""Tool-argument recording and diagnostics in the agent eval."""

from __future__ import annotations

import json
from pathlib import Path

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
    rendered = _tool_call_record(record)

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


def test_related_context_reports_the_flag_recorded_at_decision_time():
    """The report repeats what was captured at dispatch and never recomputes it."""
    for shown in (True, False, None):
        record = ToolCallRecord(
            tool_name="get_related_context",
            args={"chunk_id": "real_1"},
            chunk_id_shown_in_evidence=shown,
        )
        assert _tool_call_record(record)["chunk_id_from_evidence"] is shown


def test_diagnostics_keep_missing_provenance_apart_from_a_negative():
    """Check: diagnostics keep missing provenance apart from a negative."""
    records = [
        _row(
            [
                _call("get_related_context", {"chunk_id": "a"}, chunk_id_from_evidence=True),
                _call("get_related_context", {"chunk_id": "b"}, chunk_id_from_evidence=False),
                _call("get_related_context", {}, chunk_id_from_evidence=None),
            ],
            {},
        )
    ]
    result = _tool_argument_diagnostics(records)["tool_argument_diagnostics"]

    assert result["related_context_calls"] == 3
    assert result["related_context_chunk_id_from_evidence"] == 1
    assert result["related_context_chunk_id_provenance_missing"] == 1


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


_EXECUTED_TEXT = ["CASE-1 status was changed from open to in_progress"]
_EXPECTED_WRITE = {"update_case_status": {"case_id": "CASE-1", "new_status": "in_progress"}}


def _write_row(case_id, new_status, *, expected_outcome="executed", evidence=None, expected=None):
    return _row(
        [_call("update_case_status", {"case_id": case_id, "new_status": new_status})],
        _EXPECTED_WRITE if expected is None else expected,
        expected_case_action_outcome=expected_outcome,
        mcp_evidence_texts=_EXECUTED_TEXT if evidence is None else evidence,
    )


def _mutation_counts(row):
    result = _tool_argument_diagnostics([row])["tool_argument_diagnostics"]
    return {
        name: result[name]
        for name in (
            "wrong_mutation_executions",
            "unexpected_mutation_executions",
            "wrong_target_mutations",
            "wrong_transition_mutations",
            "unverified_mutation_executions",
        )
    }


def test_correct_executed_write_counts_as_no_wrong_mutation():
    """Check: correct executed write counts as no wrong mutation."""
    assert not any(_mutation_counts(_write_row("CASE-1", "in_progress")).values())


def test_executed_write_on_the_wrong_case_is_a_wrong_mutation():
    """A row expecting a write can no longer hide a write to a different case."""
    counts = _mutation_counts(_write_row("CASE-9", "in_progress"))
    assert counts["wrong_mutation_executions"] == 1
    assert counts["wrong_target_mutations"] == 1


def test_executed_write_with_the_wrong_status_is_a_wrong_mutation():
    """Check: executed write with the wrong status is a wrong mutation."""
    counts = _mutation_counts(_write_row("CASE-1", "resolved"))
    assert counts["wrong_mutation_executions"] == 1
    assert counts["wrong_transition_mutations"] == 1


def test_denied_write_is_not_a_mutation():
    """A call whose evidence shows no execution changed nothing."""
    row = _write_row(
        "CASE-9", "in_progress", evidence=["requires approval before it can be applied"]
    )
    assert not any(_mutation_counts(row).values())


def test_write_on_a_row_that_expected_none_is_unexpected():
    """Check: write on a row that expected none is unexpected."""
    counts = _mutation_counts(_write_row("CASE-1", "in_progress", expected_outcome=None))
    assert counts["wrong_mutation_executions"] == 1
    assert counts["unexpected_mutation_executions"] == 1


def test_executed_write_without_gold_arguments_is_unverified_not_wrong():
    """Missing gold cannot prove a write wrong, so it is reported on its own."""
    counts = _mutation_counts(_write_row("CASE-1", "in_progress", expected={}))
    assert counts["unverified_mutation_executions"] == 1
    assert counts["wrong_mutation_executions"] == 0


def test_executed_action_gold_row_is_valid_and_kept_apart_from_the_benchmark():
    """The mutating row parses, is checkable, and stays out of the 14-row benchmark file."""
    data = Path(__file__).resolve().parents[2] / "data/eval"
    lines = (data / "specialized_tool_executed_action_gold.jsonl").read_text("utf-8").splitlines()
    assert len(lines) == 1
    example = GoldExample(**json.loads(lines[0]))

    assert example.expected_case_action_outcome == "executed"
    assert example.expected_tool_args == {
        "update_case_status": {"case_id": "CASE-2002", "new_status": "in_progress"}
    }
    benchmark = (data / "specialized_tool_gold.jsonl").read_text("utf-8").splitlines()
    assert all(json.loads(line)["expected_case_action_outcome"] != "executed" for line in benchmark)
