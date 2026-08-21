from __future__ import annotations

from datetime import UTC, datetime

from rag.agent.state import AgentState
from rag.config import load_config
from rag.eval.gold_schema import GoldExample
from rag.eval.run_agent_eval import (
    _cited_sources,
    _content_by_source,
    _extract_cited_source_numbers,
    _infer_cited_sources,
    _resolve_citation_attribution,
    evaluate_agent,
)
from rag.schemas import Chunk, ChunkMetadata, SearchResult


def _chunk() -> Chunk:
    """Build a single Chunk pointing at the rollback runbook, matching the gold example below."""
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id="doc-1",
        chunk_id="doc-1_0",
        source="knowledge_base/runbooks/failed-kubernetes-deployment.md",
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="techfusion",
    )
    return Chunk(id="doc-1_0", content="Rollback to the last known-good digest.", metadata=metadata)


class ScriptedLLM:
    """LLM double that returns each response in order."""

    def __init__(self, responses: list[str]) -> None:
        """Store the queued responses this double's generate() will pop from."""
        self._responses = list(responses)

    def generate(self, system: str, user: str) -> str:
        """Return the next queued response."""
        return self._responses.pop(0)

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


class FakePipeline:
    """RetrievalPipeline double recording answer()/retrieve() call counts."""

    def __init__(self) -> None:
        """Start with zero answer()/retrieve() call counts."""
        self.answer_calls = 0
        self.retrieve_calls = 0

    def answer(self, query, filters=None, candidate_k=None, auth=None):
        """Record the call and return a fixed classic-RAG answer."""
        self.answer_calls += 1
        return {
            "answer": "classic answer",
            "sources": [],
            "retrieval_ms": 1.0,
            "generation_ms": 1.0,
            "total_ms": 2.0,
        }

    def retrieve(self, query, filters=None, candidate_k=None, auth=None):
        """Record the call and return one fixed result."""
        self.retrieve_calls += 1
        return [SearchResult(chunk=_chunk(), score=0.9)]

    def sanitize_evidence(self, results, auth):
        """Return results unchanged; not exercised by these tests."""
        return results


class FakeVectorStore:
    """Minimal VectorStore double that only answers health checks."""

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


class FakeEmbedder:
    """Minimal Embedder double returning fixed placeholder vectors."""

    def embed_query(self, text):
        """Return a placeholder vector."""
        return [0.0]

    def embed_documents(self, texts):
        """Return one placeholder vector per input text; unused here."""
        return [[0.0] for _ in texts]


def _agent_config():
    """Return `load_config()` with the agent enabled."""
    config = load_config().model_copy(deep=True)
    agent = config.agent.model_copy(update={"enabled": True})
    return config.model_copy(update={"agent": agent})


def test_evaluate_agent_reports_routing_and_tool_metrics():
    """evaluate_agent() aggregates routing/tool/citation metrics across a mixed gold set."""
    examples = [
        GoldExample(
            question="What is the maximum supported document size?",
            expected_answer="50 MB",
            agentic_category="tool_not_needed",
            tool_not_needed=True,
        ),
        GoldExample(
            question="Which service caused the backlog and what is the rollback?",
            expected_answer="Rollback to the last known-good digest.",
            relevant_documents=["knowledge_base/runbooks/failed-kubernetes-deployment.md"],
            agentic_category="query_decomposition",
            requires_query_decomposition=True,
            requires_multiple_retrieval_calls=True,
            expected_tool_sequence=["search_knowledge_base"],
        ),
    ]
    llm = ScriptedLLM(
        [
            # Example 1: routes classic (tool_not_needed).
            '{"query_type": "simple"}',
            # Example 2: full agent loop.
            '{"query_type": "complex"}',
            '{"subquestions": ["which service"]}',
            '{"tool_name": "search_knowledge_base", "tool_args": {"query": "which service"}}',
            '{"sufficient": true}',
            "Rollback to the last known-good digest. (Source 1)",
        ]
    )
    pipeline = FakePipeline()
    config = _agent_config()

    report = evaluate_agent(
        pipeline, FakeVectorStore(), FakeEmbedder(), llm, examples, "techfusion", config
    )

    assert report["num_examples"] == 2
    assert report["routing_accuracy"]["count"] == 2
    assert report["routing_accuracy"]["rate"] == 1.0
    assert report["unnecessary_agent_rate"]["count"] == 1
    assert report["unnecessary_agent_rate"]["rate"] == 0.0
    assert report["tool_selection_accuracy"]["count"] == 1
    assert report["tool_selection_accuracy"]["rate"] == 1.0
    assert report["average_tool_calls"]["count"] == 1  # only the agent-routed example
    assert report["average_tool_calls"]["value"] == 1.0
    assert report["citation_support_rate"]["count"] == 1
    assert report["citation_support_rate"]["rate"] == 1.0
    assert report["by_agentic_category"]["tool_not_needed"]["count"] == 1
    assert report["by_agentic_category"]["query_decomposition"]["count"] == 1
    assert pipeline.answer_calls == 1
    assert pipeline.retrieve_calls == 1


def test_evaluate_agent_answer_correctness_and_verbose_output():
    """verbose=True adds a per_example breakdown alongside the aggregate correctness metric."""
    examples = [
        GoldExample(
            question="What is the maximum supported document size?",
            expected_answer="fifty megabytes maximum document size",
            tool_not_needed=True,
        ),
    ]
    llm = ScriptedLLM(['{"query_type": "simple"}'])
    pipeline = FakePipeline()

    report = evaluate_agent(
        pipeline,
        FakeVectorStore(),
        FakeEmbedder(),
        llm,
        examples,
        "techfusion",
        _agent_config(),
        verbose=True,
    )

    assert report["agent_answer_correctness"]["count"] == 1
    assert "per_example" in report
    assert report["per_example"][0]["route"] == "classic_rag"


def test_extract_cited_source_numbers_parses_common_citation_phrasings():
    """Parses "(Source N)", "Sources N and M", and reports no citation as an empty set."""
    assert _extract_cited_source_numbers("The answer is X (Source 2).") == {2}
    assert _extract_cited_source_numbers("As stated in Sources 1 and 3.") == {1, 3}
    assert _extract_cited_source_numbers("No citation here at all.") == set()


def test_cited_sources_returns_none_when_nothing_is_cited():
    """None (not an empty list) distinguishes 'cited nothing identifiable' from 'cited zero'."""
    assert _cited_sources("No source mention.", ["a.md", "b.md"]) is None


def test_cited_sources_maps_1_indexed_numbers_to_citation_order():
    """A parsed 'Source N' maps to citations[N-1]; an out-of-range number resolves to None.

    None (not an empty list) so `_resolve_citation_attribution` still
    falls back to keyword-overlap inference for an answer that mentions
    "Source" but only with numbers that don't resolve to anything --
    treated the same as "no citation parsed", not as "explicitly cited
    zero sources".
    """
    assert _cited_sources("See (Source 2).", ["a.md", "b.md"]) == ["b.md"]
    assert _cited_sources("See (Source 9).", ["a.md", "b.md"]) is None


def test_infer_cited_sources_matches_on_keyword_overlap():
    """A source whose content shares >=3 words (len>3) with the answer is inferred cited."""
    content_by_source = {
        "a.md": "Rollback to the last known-good digest immediately.",
        "b.md": "Unrelated architecture overview text about something else.",
    }
    inferred = _infer_cited_sources(
        "Operators should rollback to the last known-good digest.", content_by_source
    )
    assert inferred == ["a.md"]


def test_infer_cited_sources_returns_empty_when_no_overlap_meets_threshold():
    """Below the overlap threshold, nothing is inferred. a genuine 'no signal' case."""
    content_by_source = {
        "a.md": "Rollback to the last known-good digest immediately.",
        "b.md": "Unrelated architecture overview text about something else.",
    }
    assert _infer_cited_sources("I have no information to share here.", content_by_source) == []


def test_resolve_citation_attribution_prefers_explicit_over_inferred():
    """An explicit '(Source N)' mention wins even when its content also overlaps another source."""

    class _FakeResult:
        route = "classic_rag"
        classic_sources = [
            {"source": "a.md", "content": "Rollback to the last known-good digest."},
            {"source": "b.md", "content": "Unrelated architecture overview text."},
        ]

    sources, attribution = _resolve_citation_attribution(
        "Rollback to the last known-good digest. (Source 1)",
        ["a.md", "b.md"],
        _FakeResult(),
        AgentState(original_query="q"),
    )
    assert sources == ["a.md"]
    assert attribution == "explicit"


def test_resolve_citation_attribution_falls_back_to_inference_when_uncited():
    """No explicit citation, but content overlap identifies the actually-used source."""

    class _FakeResult:
        route = "classic_rag"
        classic_sources = [
            {"source": "a.md", "content": "Rollback to the last known-good digest."},
            {"source": "b.md", "content": "Unrelated architecture overview text."},
        ]

    sources, attribution = _resolve_citation_attribution(
        "Rollback to the last known-good digest.",
        ["a.md", "b.md"],
        _FakeResult(),
        AgentState(original_query="q"),
    )
    assert sources == ["a.md"]
    assert attribution == "inferred"


def test_resolve_citation_attribution_reports_none_when_nothing_matches():
    """Neither explicit citation nor content overlap. honestly reported as unattributed."""

    class _FakeResult:
        route = "classic_rag"
        classic_sources = [
            {"source": "a.md", "content": "Rollback to the last known-good digest."},
        ]

    sources, attribution = _resolve_citation_attribution(
        "I have no information to share here.",
        ["a.md"],
        _FakeResult(),
        AgentState(original_query="q"),
    )
    assert sources is None
    assert attribution == "none"


def test_content_by_source_reads_agent_route_from_retrieved_evidence():
    """On the agent route, content comes from state.retrieved_evidence, not classic_sources."""
    now = datetime.now(UTC)
    chunk = Chunk(
        id="c_0",
        content="Rollback content here.",
        metadata=ChunkMetadata(
            document_id="c",
            chunk_id="c_0",
            source="a.md",
            source_type="text",
            created_at=now,
            last_modified=now,
            chunk_index=0,
            dataset_id="techfusion",
        ),
    )
    state = AgentState(original_query="q")
    state.retrieved_evidence = [SearchResult(chunk=chunk, score=0.9)]

    class _FakeResult:
        route = "agent"
        classic_sources: list = []

    content = _content_by_source(_FakeResult(), state)
    assert content == {"a.md": "Rollback content here."}


def _two_chunk() -> tuple[Chunk, Chunk]:
    """Build a relevant chunk and a tangential one, matching only relevant_documents."""
    now = datetime.now(UTC)
    relevant = Chunk(
        id="relevant_0",
        content="Rollback to the last known-good digest.",
        metadata=ChunkMetadata(
            document_id="relevant",
            chunk_id="relevant_0",
            source="knowledge_base/runbooks/failed-kubernetes-deployment.md",
            source_type="text",
            created_at=now,
            last_modified=now,
            chunk_index=0,
            dataset_id="techfusion",
        ),
    )
    tangential = Chunk(
        id="tangential_0",
        content="Unrelated architecture overview text.",
        metadata=ChunkMetadata(
            document_id="tangential",
            chunk_id="tangential_0",
            source="knowledge_base/architecture/system-overview.md",
            source_type="text",
            created_at=now,
            last_modified=now,
            chunk_index=0,
            dataset_id="techfusion",
        ),
    )
    return relevant, tangential


class TwoChunkPipeline:
    """RetrievalPipeline double returning a relevant + a tangential chunk from one call."""

    def __init__(self) -> None:
        """Start with a zero retrieve() call count."""
        self.retrieve_calls = 0

    def retrieve(self, query, filters=None, candidate_k=None, auth=None):
        """Record the call and return one relevant chunk plus one tangential chunk."""
        self.retrieve_calls += 1
        relevant, tangential = _two_chunk()
        return [
            SearchResult(chunk=relevant, score=0.9),
            SearchResult(chunk=tangential, score=0.5),
        ]

    def sanitize_evidence(self, results, auth):
        """Return results unchanged; not exercised by these tests."""
        return results


def test_citation_support_rate_scores_only_answer_cited_sources():
    """Regression test: a tangential, gathered-but-uncited chunk must not fail grounding.

    Both chunks are accumulated into state.retrieved_evidence (Source 1 =
    relevant, Source 2 = tangential), but the synthesized answer only
    cites "(Source 1)". Under the pre-fix definition (score every gathered
    chunk), this example would fail because the tangential Source 2 never
    matches relevant_documents. Under the fix, only the actually-cited
    Source 1 is scored, and it does match.
    """
    examples = [
        GoldExample(
            question="Which service was rolled back and what is the rollback rule?",
            expected_answer="Rollback to the last known-good digest.",
            relevant_documents=["knowledge_base/runbooks/failed-kubernetes-deployment.md"],
            agentic_category="query_decomposition",
            requires_query_decomposition=True,
            requires_multiple_retrieval_calls=True,
        ),
    ]
    llm = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subquestions": ["which service"]}',
            '{"tool_name": "search_knowledge_base", "tool_args": {"query": "which service"}}',
            '{"sufficient": true}',
            "Rollback to the last known-good digest. (Source 1)",
        ]
    )
    pipeline = TwoChunkPipeline()

    report = evaluate_agent(
        pipeline, FakeVectorStore(), FakeEmbedder(), llm, examples, "techfusion", _agent_config()
    )

    assert report["citation_support_rate"]["count"] == 1
    assert report["citation_support_rate"]["rate"] == 1.0
    assert report["citation_support_rate"]["uncited_answer_count"] == 0


def test_citation_support_rate_infers_a_citation_when_uncited_but_content_overlaps():
    """No explicit "(Source N)", but the answer echoes the relevant chunk's own wording.

    Regression test for the citation-compliance finding (qwen2.5:3b often
    skips the requested citation format even when grounded correctly --
    see ISSUES.md): the keyword-overlap fallback recovers a correct
    attribution here instead of discarding this example as unscoreable.
    """
    examples = [
        GoldExample(
            question="Which service was rolled back and what is the rollback rule?",
            expected_answer="Rollback to the last known-good digest.",
            relevant_documents=["knowledge_base/runbooks/failed-kubernetes-deployment.md"],
            agentic_category="query_decomposition",
            requires_query_decomposition=True,
            requires_multiple_retrieval_calls=True,
        ),
    ]
    llm = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subquestions": ["which service"]}',
            '{"tool_name": "search_knowledge_base", "tool_args": {"query": "which service"}}',
            '{"sufficient": true}',
            "Rollback to the last known-good digest.",  # no "Source N" mention
        ]
    )
    pipeline = TwoChunkPipeline()

    report = evaluate_agent(
        pipeline, FakeVectorStore(), FakeEmbedder(), llm, examples, "techfusion", _agent_config()
    )

    assert report["citation_support_rate"]["count"] == 1
    assert report["citation_support_rate"]["rate"] == 1.0
    assert report["citation_support_rate"]["explicit_count"] == 0
    assert report["citation_support_rate"]["inferred_count"] == 1
    assert report["citation_support_rate"]["uncited_answer_count"] == 0


def test_citation_support_rate_excludes_answers_with_no_attribution_signal_at_all():
    """An answer with neither an explicit citation nor content overlap is truly unscoreable."""
    examples = [
        GoldExample(
            question="Which service was rolled back and what is the rollback rule?",
            expected_answer="Rollback to the last known-good digest.",
            relevant_documents=["knowledge_base/runbooks/failed-kubernetes-deployment.md"],
            agentic_category="query_decomposition",
            requires_query_decomposition=True,
            requires_multiple_retrieval_calls=True,
        ),
    ]
    llm = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subquestions": ["which service"]}',
            '{"tool_name": "search_knowledge_base", "tool_args": {"query": "which service"}}',
            '{"sufficient": true}',
            "I have no information to share here.",  # no citation, no content overlap
        ]
    )
    pipeline = TwoChunkPipeline()

    report = evaluate_agent(
        pipeline, FakeVectorStore(), FakeEmbedder(), llm, examples, "techfusion", _agent_config()
    )

    assert report["citation_support_rate"]["count"] == 0
    assert report["citation_support_rate"]["rate"] is None
    assert report["citation_support_rate"]["uncited_answer_count"] == 1


def test_tool_selection_coverage_supplements_the_strict_accuracy_gate():
    """A partial-but-sensible tool match reads 0.0 strict but > 0 on the graded metrics.

    Gold expects 2 tools; the agent only calls 1 (which is in the expected
    set). tool_selection_accuracy is 0.0 (strict all-or-nothing), but
    required_tool_coverage is 0.5 and expected_tool_precision is 1.0.
    """
    examples = [
        GoldExample(
            question="Which service was rolled back and what is the rollback rule?",
            expected_answer="Rollback to the last known-good digest.",
            relevant_documents=["knowledge_base/runbooks/failed-kubernetes-deployment.md"],
            agentic_category="query_decomposition",
            requires_query_decomposition=True,
            requires_multiple_retrieval_calls=True,
            expected_tool_sequence=["search_knowledge_base", "get_document"],
        ),
    ]
    llm = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subquestions": ["which service"]}',
            '{"tool_name": "search_knowledge_base", "tool_args": {"query": "which service"}}',
            '{"sufficient": true}',
            "Rollback to the last known-good digest. (Source 1)",
        ]
    )
    pipeline = TwoChunkPipeline()

    report = evaluate_agent(
        pipeline, FakeVectorStore(), FakeEmbedder(), llm, examples, "techfusion", _agent_config()
    )

    assert report["tool_selection_accuracy"]["rate"] == 0.0
    coverage = report["tool_selection_coverage"]
    assert coverage["required_tool_coverage"]["mean"] == 0.5
    assert coverage["expected_tool_precision"]["mean"] == 1.0
    assert coverage["unexpected_tool_rate"]["mean"] == 0.0
