"""Regression test for CTX-006: `_synthesize` must budget its own system template.

`_synthesize` renders the final answer with the agent's own
`agent_synthesize_v2.yaml` template, not `RetrievalPipeline`'s classic-RAG
`generation.prompt` template. `build_context_text` defaults to measuring
the *pipeline's* own system template for budgeting, so `_synthesize` must
explicitly override that default with its own template's
`system_template`; otherwise the Context Engineering budget is computed
against the wrong system prompt's token cost for every agent-routed
synthesis call.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rag.agent.graph import _synthesize
from rag.agent.state import AgentState
from rag.prompts.loader import PromptTemplate
from rag.schemas import Chunk, ChunkMetadata, SearchResult


def _result(chunk_id: str, content: str) -> SearchResult:
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id=chunk_id,
        chunk_id=chunk_id,
        source=f"{chunk_id}.md",
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="test-dataset",
    )
    return SearchResult(chunk=Chunk(id=chunk_id, content=content, metadata=metadata), score=0.9)


class _CapturingPipeline:
    """RetrievalPipeline double that records build_context_text's exact kwargs."""

    def __init__(self) -> None:
        self.captured_system_prompt_text: str | None = "NOT_CALLED"
        self.prompt_template_system_template = "CLASSIC RAG SYSTEM PROMPT"

    def build_context_text(
        self, results, query, *, history_turns=None, tenant_id=None, system_prompt_text=None
    ):
        self.captured_system_prompt_text = system_prompt_text
        return "rendered context", None, results


class _FakeLLM:
    def generate(self, system: str, user: str) -> str:
        return "the answer"

    def health_check(self) -> bool:
        return True


def test_synthesize_passes_its_own_templates_system_template_for_budgeting():
    """build_context_text must receive the synthesize template's own system_template."""
    synthesize_template = PromptTemplate(
        prompt_id="agent_synthesize",
        version="v2-test",
        description="test",
        system_template="AGENT SYNTHESIZE SYSTEM PROMPT",
        user_template="Evidence:\n{context}\n\nQuestion: {query}",
        required_variables=["context", "query"],
        created_at="2026-01-01",
    )
    state = AgentState(original_query="q", retrieved_evidence=[_result("c1", "Alpha content.")])
    pipeline = _CapturingPipeline()

    _synthesize(state, _FakeLLM(), synthesize_template, pipeline)

    assert pipeline.captured_system_prompt_text == "AGENT SYNTHESIZE SYSTEM PROMPT"
    assert pipeline.captured_system_prompt_text != pipeline.prompt_template_system_template
