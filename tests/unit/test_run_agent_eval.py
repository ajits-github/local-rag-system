from __future__ import annotations

from datetime import UTC, datetime

from rag.config import load_config
from rag.eval.gold_schema import GoldExample
from rag.eval.run_agent_eval import evaluate_agent
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
