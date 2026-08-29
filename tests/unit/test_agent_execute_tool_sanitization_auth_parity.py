"""Tests that _execute_tool sanitizes evidence with resolved auth.

Direct-fetch tools resolve `auth` before calling VectorStore. The graph's
`_execute_tool` node must pass the same resolved context to
`pipeline.sanitize_evidence(...)` so field redaction has the same
effective authorization context for every tool path.

Uses a real `RetrievalPipeline` (fakes for vectorstore/embedder, no
Postgres/Ollama) so the real `resolve_auth`/`sanitize_evidence` logic is
exercised end to end through `run_agent()`, not a test double's
approximation of it.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rag.agent.graph import run_agent
from rag.agent.state import AgentState
from rag.config import load_config
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.pipeline import RetrievalPipeline
from rag.schemas import Chunk, ChunkMetadata

_SECRET = "SYNTHETIC_ONLY_abc123"
_SOURCE = "policy.md"
_CHUNK_ID = "doc-1_0"


def _chunk() -> Chunk:
    """Build the one sensitive-field-tagged chunk every fake vectorstore method returns."""
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id="doc-1",
        chunk_id=_CHUNK_ID,
        source=_SOURCE,
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="ds1",
        sensitive_field_ids=["synthetic_admin_credential"],
    )
    return Chunk(id=_CHUNK_ID, content=f"The admin token is {_SECRET}.", metadata=metadata)


class FakeVectorStore:
    """Serves the same sensitive chunk via both the dense-search and get_chunks_by_source paths."""

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True

    def search(self, query_embedding, top_k, filters=None, auth=None):
        """Return the one fixed chunk as a dense search hit, ignoring the embedding."""
        from rag.schemas import SearchResult

        return [SearchResult(chunk=_chunk(), score=0.9)]

    def search_keyword(self, query, top_k, filters=None, auth=None):
        """Unused: config.retrieval.provider is dense in these tests."""
        return []

    def get_chunks_by_source(self, source, dataset_id, auth=None, limit=None):
        """Return the one fixed chunk, ignoring source/dataset_id/limit."""
        return [_chunk()]

    def list_document_versions(self, dataset_id):
        """No versions; freshness resolution is not what these tests probe."""
        return []


class FakeEmbedder:
    """Placeholder embedder; the fake vectorstore ignores the actual vector."""

    def embed_query(self, text: str) -> list[float]:
        """Return a placeholder vector."""
        return [0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one placeholder vector per input text; unused here."""
        return [[0.0] for _ in texts]


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


def _config():
    """Return a config with authorization off, field redaction on, and the agent enabled."""
    config = load_config().model_copy(deep=True)
    config.security.authorization.enabled = False
    config.security.field_redaction.enabled = True
    config.agent.enabled = True
    return config


def _run(tool_call_json: str) -> str:
    """Run one agent turn dispatching a single tool call, returning the gathered evidence content.

    The caller asserts an "admin" role that would authorize disclosure of
    the tagged field if it were used raw. `authorization.enabled=False`
    means the correct, fail-closed behavior is to redact regardless.
    """
    llm = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subquestions": ["q1"]}',
            tool_call_json,
            '{"sufficient": true}',
            "final answer",
        ]
    )
    vectorstore = FakeVectorStore()
    embedder = FakeEmbedder()
    pipeline = RetrievalPipeline(_config(), vectorstore=vectorstore, embedder=embedder)
    caller_with_admin_role = AuthorizationContext(
        tenant_id="tenant_alpha", roles=["tenant_alpha_admin"]
    )
    state = AgentState(
        original_query="what is the admin token",
        authorization_context=caller_with_admin_role,
        filters={"dataset_id": "ds1"},
    )

    result = run_agent(
        state,
        pipeline=pipeline,
        vectorstore=vectorstore,
        embedder=embedder,
        llm=llm,
        config=_config(),
    )

    assert len(result.state.retrieved_evidence) == 1
    return result.state.retrieved_evidence[0].chunk.content


def test_get_document_result_fails_closed_when_authorization_disabled_despite_caller_role():
    """A direct-fetch tool's evidence redacts the sensitive field, matching the fail-closed rule.

    The caller asserts a role that would otherwise be authorized for this
    field, so the test proves disabled authorization resolves to
    fail-closed field redaction.
    """
    content = _run(f'{{"tool_name": "get_document", "tool_args": {{"source": "{_SOURCE}"}}}}')

    assert _SECRET not in content
    assert "[REDACTED:SENSITIVE_FIELD]" in content


def test_get_document_and_search_knowledge_base_redact_identically():
    """Direct-tool and search_knowledge_base evidence are redacted the same way here."""
    direct_content = _run(
        f'{{"tool_name": "get_document", "tool_args": {{"source": "{_SOURCE}"}}}}'
    )
    search_content = _run(
        '{"tool_name": "search_knowledge_base", "tool_args": {"query": "admin token"}}'
    )

    assert direct_content == search_content
    assert _SECRET not in direct_content
    assert _SECRET not in search_content


def test_get_document_preserves_the_field_when_authorization_enabled_and_role_is_authorized():
    """Sanity check: authorization.enabled=True with a genuinely authorized role still discloses.

    The fail-closed case above is specific to authorization being
    disabled; authorized roles still receive the unredacted field.
    """
    config = _config()
    config.security.authorization.enabled = True
    llm = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subquestions": ["q1"]}',
            f'{{"tool_name": "get_document", "tool_args": {{"source": "{_SOURCE}"}}}}',
            '{"sufficient": true}',
            "final answer",
        ]
    )
    vectorstore = FakeVectorStore()
    embedder = FakeEmbedder()
    pipeline = RetrievalPipeline(config, vectorstore=vectorstore, embedder=embedder)
    caller_with_admin_role = AuthorizationContext(
        tenant_id="tenant_alpha", roles=["tenant_alpha_admin"]
    )
    state = AgentState(
        original_query="what is the admin token",
        authorization_context=caller_with_admin_role,
        filters={"dataset_id": "ds1"},
    )

    result = run_agent(
        state, pipeline=pipeline, vectorstore=vectorstore, embedder=embedder, llm=llm, config=config
    )

    assert _SECRET in result.state.retrieved_evidence[0].chunk.content
