"""Bounded handling of an exact repeat tool proposal in `run_agent`."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from rag.agent.graph import run_agent
from rag.agent.state import AgentState
from rag.config import load_config
from rag.schemas import Chunk, ChunkMetadata, SearchResult

_V7 = "src/rag/prompts/templates/agent_tool_select_v7.yaml"
_V3 = "src/rag/prompts/templates/agent_tool_select_v3.yaml"
_SEARCH = {"tool_name": "search_knowledge_base", "tool_args": {"query": "same"}}
_INSUFFICIENT = {"sufficient": False, "reformulated_query": None, "reasoning": "more"}
_SUFFICIENT = {"sufficient": True, "reformulated_query": None, "reasoning": "enough"}


class _ScriptedLLM:
    """Reply from a fixed list and record every call."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.users: list[str] = []

    def generate(self, system: str, user: str) -> str:
        """Return the next scripted reply."""
        self.users.append(user)
        reply = self.replies.pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)

    def health_check(self) -> bool:
        """Report healthy."""
        return True


def _result() -> SearchResult:
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id="doc",
        chunk_id="doc_1",
        source="a.md",
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="ds",
    )
    return SearchResult(chunk=Chunk(id="doc_1", content="body", metadata=metadata), score=1.0)


def _run(
    replies: list,
    tool_select_path: str,
    *,
    max_steps: int = 8,
    cancel_event=None,
    retrieve_side_effect=None,
    max_summary_chars: int | None = None,
):
    config = load_config().model_copy(deep=True)
    config.agent.enabled = True
    config.agent.max_agent_steps = max_steps
    config.agent.tool_select_prompt_path = tool_select_path
    config.agent.classify_prompt_path = "src/rag/prompts/templates/agent_classify_v3.yaml"
    config.agent.decompose_prompt_path = "src/rag/prompts/templates/agent_decompose_v2.yaml"
    config.mcp.client.enabled = False
    config.graph.enabled = False
    pipeline = MagicMock()
    pipeline.retrieve.return_value = [_result()]
    if retrieve_side_effect is not None:
        pipeline.retrieve.side_effect = retrieve_side_effect
    if max_summary_chars is not None:
        config.agent.max_evidence_summary_chars = max_summary_chars
    pipeline.sanitize_evidence.side_effect = lambda results, auth: results
    pipeline.build_context_text.return_value = ("ctx", None, [_result()])
    llm = _ScriptedLLM(replies)
    state = AgentState(original_query="question", filters={"dataset_id": "ds"})
    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=MagicMock(),
        embedder=MagicMock(),
        llm=llm,
        config=config,
        cancel_event=cancel_event,
    )
    return result, pipeline, llm


_PREFIX = [{"query_type": "complex", "reasoning": "r"}, {"subquestions": ["question"]}]


def test_identical_search_is_rejected_then_the_corrected_tool_runs():
    """A repeat is not dispatched, and the model is told why and may pick another tool."""
    related = {"tool_name": "get_related_context", "tool_args": {"chunk_id": "doc_1"}}
    replies = [*_PREFIX, _SEARCH, _INSUFFICIENT, _SEARCH, related, _INSUFFICIENT, "answer"]

    result, pipeline, llm = _run(replies, _V7, max_steps=9)

    assert [r.tool_name for r in result.state.tool_call_history] == [
        "search_knowledge_base",
        "get_related_context",
    ]
    assert pipeline.retrieve.call_count == 1
    assert result.state.repeated_tool_proposals == 1
    assert any("repeated one of these calls exactly" in user for user in llm.users)


def test_second_repeat_stops_the_loop_and_answers_from_gathered_evidence():
    """Two repeats end the loop with a normal synthesis, never an extra dispatch."""
    replies = [*_PREFIX, _SEARCH, _INSUFFICIENT, _SEARCH, _SEARCH, "answer"]

    result, pipeline, _ = _run(replies, _V7)

    assert pipeline.retrieve.call_count == 1
    assert result.state.repeated_tool_proposals == 2
    assert result.state.termination_reason == "synthesized"
    assert result.state.final_answer == "answer"


def test_older_templates_keep_their_recorded_behavior():
    """A template without tool history still dispatches an identical repeat."""
    replies = [*_PREFIX, _SEARCH, _INSUFFICIENT, _SEARCH, _INSUFFICIENT]

    result, pipeline, _ = _run(replies, _V3)

    assert pipeline.retrieve.call_count == 2
    assert result.state.repeated_tool_proposals == 0


_RELATED = {"tool_name": "get_related_context", "tool_args": {"chunk_id": "doc_1"}}


def _nodes_excluding_synthesis(result) -> int:
    timings = result.state.node_timings_ms
    return sum(len(v) for name, v in timings.items() if name != "synthesize")


def test_corrective_selection_at_the_ceiling_does_not_dispatch():
    """The corrective selection spends the last step, so its tool never runs."""
    replies = [*_PREFIX, _SEARCH, _INSUFFICIENT, _SEARCH, _RELATED, "answer"]

    result, pipeline, _ = _run(replies, _V7, max_steps=7)

    assert [r.tool_name for r in result.state.tool_call_history] == ["search_knowledge_base"]
    assert result.state.termination_reason == "max_steps"
    assert result.state.step_count == 7
    assert _nodes_excluding_synthesis(result) == 7


def test_step_count_matches_nodes_at_the_default_ceiling():
    """At eight steps the corrected tool runs, and no evidence check runs past the bound."""
    replies = [*_PREFIX, _SEARCH, _INSUFFICIENT, _SEARCH, _RELATED, "answer"]

    result, _, _ = _run(replies, _V7, max_steps=8)

    assert [r.tool_name for r in result.state.tool_call_history] == [
        "search_knowledge_base",
        "get_related_context",
    ]
    assert result.state.step_count == 8
    assert _nodes_excluding_synthesis(result) == 8
    assert result.state.termination_reason == "max_steps"


def test_step_count_matches_nodes_with_budget_to_spare():
    """The control: with room left, the counter still equals the nodes that ran."""
    replies = [*_PREFIX, _SEARCH, _INSUFFICIENT, _SEARCH, _RELATED, _SUFFICIENT, "answer"]

    result, _, _ = _run(replies, _V7, max_steps=12)

    assert len(result.state.tool_call_history) == 2
    assert result.state.step_count == _nodes_excluding_synthesis(result) == 9
    assert result.state.termination_reason == "synthesized"


def test_second_repeat_at_the_ceiling_is_labeled_max_steps():
    """A corrective selection that exhausts the budget ends the run before the repeat check."""
    replies = [*_PREFIX, _SEARCH, _INSUFFICIENT, _SEARCH, _SEARCH, "answer"]

    result, pipeline, _ = _run(replies, _V7, max_steps=7)

    assert pipeline.retrieve.call_count == 1
    assert result.state.repeated_tool_proposals == 1
    assert result.state.termination_reason == "max_steps"


def test_cancellation_during_corrective_selection_stops_before_dispatch():
    """A caller gone by the end of the corrective selection never gets the tool run."""
    cancel = threading.Event()

    class _CancelOnCorrection(list):
        """Reply list that sets the cancel event once the corrected proposal is popped."""

        def pop(self, index=-1):
            reply = super().pop(index)
            if reply is _RELATED:
                cancel.set()
            return reply

    replies = [*_PREFIX, _SEARCH, _INSUFFICIENT, _SEARCH, _RELATED, "answer"]
    llm = _ScriptedLLM([])
    llm.replies = _CancelOnCorrection(replies)

    result = _run_with_llm(llm, cancel)

    assert [r.tool_name for r in result.state.tool_call_history] == ["search_knowledge_base"]
    assert result.state.termination_reason == "cancelled"


def _run_with_llm(llm, cancel_event):
    with patch(f"{__name__}._ScriptedLLM", return_value=llm):
        result, _, _ = _run([], _V7, max_steps=12, cancel_event=cancel_event)
    return result


def _results(*chunk_ids: str) -> list[SearchResult]:
    results = []
    for chunk_id in chunk_ids:
        result = _result().model_copy(deep=True)
        result.chunk.metadata.chunk_id = chunk_id
        result.chunk.content = "x" * 400
        results.append(result)
    return results


def test_chunk_id_shown_is_recorded_when_the_model_saw_it():
    """A prior search that displayed the id counts as grounded."""
    replies = [*_PREFIX, _SEARCH, _INSUFFICIENT, _RELATED, _SUFFICIENT, "answer"]

    result, _, _ = _run(replies, _V7, max_steps=12)

    assert result.state.tool_call_history[0].chunk_id_shown_in_evidence is None
    assert result.state.tool_call_history[1].chunk_id_shown_in_evidence is True


def test_chunk_id_that_appears_only_later_stays_ungrounded():
    """Evidence gathered after the call cannot retroactively ground its argument."""
    guessed = {"tool_name": "get_related_context", "tool_args": {"chunk_id": "doc_later"}}
    later = {"tool_name": "search_knowledge_base", "tool_args": {"query": "other"}}
    replies = [*_PREFIX, _SEARCH, _INSUFFICIENT, guessed, _INSUFFICIENT, later, _SUFFICIENT, "ok"]
    calls: list[list[SearchResult]] = [_results("doc_1"), _results("doc_later")]

    result, _, _ = _run(replies, _V7, max_steps=12, retrieve_side_effect=calls)

    assert result.state.tool_call_history[1].chunk_id_shown_in_evidence is False
    assert "doc_later" in {r.chunk.metadata.chunk_id for r in result.state.retrieved_evidence}


def test_chunk_id_beyond_the_summary_cap_does_not_count_as_shown():
    """Evidence held in state but cut from the bounded summary was never shown."""
    hidden = {"tool_name": "get_related_context", "tool_args": {"chunk_id": "doc_late"}}
    replies = [*_PREFIX, _SEARCH, _INSUFFICIENT, hidden, _SUFFICIENT, "answer"]
    evidence = _results(*[f"doc_{i}" for i in range(4)], "doc_late")

    result, _, _ = _run(
        replies, _V7, max_steps=12, retrieve_side_effect=[evidence], max_summary_chars=300
    )

    record = result.state.tool_call_history[1]
    assert record.chunk_id_shown_in_evidence is False
    assert "doc_late" in {r.chunk.metadata.chunk_id for r in result.state.retrieved_evidence}


def test_invalid_related_context_arguments_have_no_provenance():
    """A rejected call is missing provenance, which is not a confirmed negative."""
    invalid = {"tool_name": "get_related_context", "tool_args": {"chunk": "doc_1"}}
    replies = [*_PREFIX, _SEARCH, _INSUFFICIENT, invalid, _SUFFICIENT, "answer"]

    result, _, _ = _run(replies, _V7, max_steps=12)

    assert result.state.tool_call_history[1].chunk_id_shown_in_evidence is None
