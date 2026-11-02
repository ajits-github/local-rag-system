"""Real-LLM checks of the agent routing, decomposition, and tool-selection prompt contracts.

Questions differ from the specialized gold set, so these test the contract and
not memorized rows. Skips when Ollama is unreachable.
"""

from __future__ import annotations

import pytest

from rag.agent.decisions import (
    ClassifyDecision,
    DecomposeDecision,
    ToolSelectionDecision,
    run_decision,
)
from rag.config import REPO_ROOT
from rag.factory import build_llm
from rag.prompts.loader import load_prompt_template

_TEMPLATES = REPO_ROOT / "src/rag/prompts/templates"
_NO_EVIDENCE = "(no evidence gathered yet)"
_NO_TOOLS = "(no tools called yet)"
_ONE_SEARCH = "- search_knowledge_base(query=first search): returned 1 result(s)"


def _template(name: str):
    return load_prompt_template(_TEMPLATES / f"{name}.yaml")


@pytest.fixture
def llm(require_ollama, config):
    """Build the model the final agent candidate uses."""
    agentic = config.model_copy(deep=True)
    agentic.generation.model_name = "qwen2.5:3b"
    return build_llm(agentic)


def _select(
    llm, template_name: str, query: str, evidence: str = _NO_EVIDENCE, history: str = _NO_TOOLS
):
    decision = run_decision(
        llm,
        _template(template_name),
        ToolSelectionDecision,
        1,
        query=query,
        original_query=query,
        tool_history=history,
        evidence_summary=evidence,
    )
    assert decision is not None, f"tool decision failed to parse for: {query!r}"
    return decision


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the maximum number of retries allowed for a failed API call?", "simple"),
        ("How many hours after account cancellation is customer data purged?", "simple"),
        ("What is the status of CASE-7001?", "complex"),
        ("Please reopen case CASE-7002.", "complex"),
        ("My old copy of the vendor policy says 60 days. What is the limit today?", "complex"),
        ("Who owns the Payments Service?", "complex"),
        ("What depends on the Ledger Database?", "complex"),
        ("Which team owns the failed service, and what is that team's escalation path?", "complex"),
    ],
)
def test_classify_v3_routes_by_capability(llm, question, expected):
    """Check: classify v3 routes by capability."""
    template = _template("agent_classify_v3")
    decision = run_decision(llm, template, ClassifyDecision, 1, query=question)
    assert decision is not None
    assert decision.query_type == expected


@pytest.mark.parametrize(
    "question",
    [
        "What is the status of CASE-7001?",
        "Please reopen case CASE-7002.",
        "Who owns the Payments Service?",
    ],
)
def test_decompose_v2_leaves_one_shot_requests_as_one_item(llm, question):
    """Check: decompose v2 leaves one shot requests as one item."""
    decision = run_decision(
        llm, _template("agent_decompose_v2"), DecomposeDecision, 1, query=question
    )
    assert decision is not None
    assert len(decision.subquestions) == 1


def test_decompose_v2_splits_a_genuinely_dependent_request(llm):
    """Check: decompose v2 splits a genuinely dependent request."""
    question = "Which service failed last night, who owns it, and what is that team's on-call rota?"
    decision = run_decision(
        llm, _template("agent_decompose_v2"), DecomposeDecision, 1, query=question
    )
    assert decision is not None
    assert 2 <= len(decision.subquestions) <= 4


def test_tool_select_searches_first_when_only_a_title_is_known(llm):
    """Check: tool select searches first when only a title is known."""
    decision = _select(
        llm,
        "agent_tool_select_v7",
        "In the Failover runbook, what do the Preconditions and Closure sections each require?",
    )
    assert decision.tool_name == "search_knowledge_base"


def test_tool_select_fetches_the_document_when_the_exact_source_is_known(llm):
    """Check: tool select fetches the document when the exact source is known."""
    evidence = (
        "[1] chunk_id=abc_1 source=docs/ops/failover.md origin=retrieved content_type=prose "
        "trust=unset: Preconditions: the replica must be in sync."
    )
    decision = _select(
        llm,
        "agent_tool_select_v7",
        "In the Failover runbook, what do the Preconditions and Closure sections each require?",
        evidence,
        _ONE_SEARCH,
    )
    assert decision.tool_name == "get_document"
    assert decision.tool_args["source"] == "docs/ops/failover.md"


def test_tool_select_resolves_an_old_exact_source_to_latest(llm):
    """Check: tool select resolves an old exact source to latest."""
    evidence = (
        "[1] chunk_id=abc_1 source=docs/hr/vendor-v1.md origin=retrieved content_type=prose "
        "trust=unset: Vendor review happens every 60 days."
    )
    decision = _select(
        llm,
        "agent_tool_select_v7",
        "My old vendor policy says 60 days. What is the current approved value?",
        evidence,
        _ONE_SEARCH,
    )
    assert decision.tool_name == "get_latest_document"
    assert decision.tool_args["source"] == "docs/hr/vendor-v1.md"


def test_tool_select_copies_the_chunk_id_for_nearby_context(llm):
    """Check: tool select copies the chunk id for nearby context."""
    evidence = (
        "[1] chunk_id=abc_77 source=docs/ops/queue.md origin=retrieved content_type=configuration "
        "trust=unset: max_batch: 40"
    )
    decision = _select(
        llm,
        "agent_tool_select_v7",
        "The queue config sets max_batch to 40. Why was that value chosen?",
        evidence,
        _ONE_SEARCH,
    )
    assert decision.tool_name == "get_related_context"
    assert decision.tool_args["chunk_id"] == "abc_77"


@pytest.mark.parametrize(
    ("query", "tool_name", "args"),
    [
        ("What priority does CASE-7001 have?", "get_case_status", {"case_id": "CASE-7001"}),
        ("Describe what CASE-7001 is about.", "get_customer_case", {"case_id": "CASE-7001"}),
        (
            "Please close case CASE-7001.",
            "update_case_status",
            {"case_id": "CASE-7001", "new_status": "closed"},
        ),
    ],
)
def test_tool_select_routes_case_requests_to_mcp(llm, query, tool_name, args):
    """Check: tool select routes case requests to mcp."""
    decision = _select(llm, "agent_tool_select_v9_mcp_actions", query)
    assert decision.tool_name == tool_name
    assert decision.tool_args == args


@pytest.mark.parametrize(
    ("query", "tool_names"),
    [
        (
            "Who owns the Payments Service?",
            {"search_knowledge_graph", "get_entity", "get_neighbors"},
        ),
        (
            "What is the dependency path from Payments Service to Ledger Database?",
            {"find_dependency_path"},
        ),
    ],
)
def test_tool_select_routes_structure_questions_to_graph(llm, query, tool_names):
    """Check: tool select routes structure questions to graph."""
    assert _select(llm, "agent_tool_select_v10_graph", query).tool_name in tool_names


def test_combined_prompt_keeps_case_and_graph_tools_reachable(llm):
    """Check: combined prompt keeps case and graph tools reachable."""
    template = "agent_tool_select_v12_mcp_actions_graph"
    assert _select(llm, template, "What priority does CASE-7001 have?").tool_name == (
        "get_case_status"
    )
    assert _select(
        llm, template, "What is the dependency path from Payments Service to Ledger Database?"
    ).tool_name == ("find_dependency_path")
