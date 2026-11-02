"""Bounded handling of an exact repeat tool proposal in `run_agent`."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

from rag.agent.graph import run_agent
from rag.agent.state import AgentState
from rag.config import load_config
from rag.schemas import Chunk, ChunkMetadata, SearchResult

_V7 = "src/rag/prompts/templates/agent_tool_select_v7.yaml"
_V3 = "src/rag/prompts/templates/agent_tool_select_v3.yaml"
_SEARCH = {"tool_name": "search_knowledge_base", "tool_args": {"query": "same"}}
_INSUFFICIENT = {"sufficient": False, "reformulated_query": None, "reasoning": "more"}


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


def _run(replies: list, tool_select_path: str):
    config = load_config().model_copy(deep=True)
    config.agent.enabled = True
    config.agent.tool_select_prompt_path = tool_select_path
    config.agent.classify_prompt_path = "src/rag/prompts/templates/agent_classify_v3.yaml"
    config.agent.decompose_prompt_path = "src/rag/prompts/templates/agent_decompose_v2.yaml"
    config.mcp.client.enabled = False
    config.graph.enabled = False
    pipeline = MagicMock()
    pipeline.retrieve.return_value = [_result()]
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
    )
    return result, pipeline, llm


_PREFIX = [{"query_type": "complex", "reasoning": "r"}, {"subquestions": ["question"]}]


def test_identical_search_is_rejected_then_the_corrected_tool_runs():
    """A repeat is not dispatched, and the model is told why and may pick another tool."""
    related = {"tool_name": "get_related_context", "tool_args": {"chunk_id": "doc_1"}}
    replies = [*_PREFIX, _SEARCH, _INSUFFICIENT, _SEARCH, related, _INSUFFICIENT, "answer"]

    result, pipeline, llm = _run(replies, _V7)

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
