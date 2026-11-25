"""Dataset/tenant scoping of chunk-id lookups (RET-013) and version lifecycle rules, on PostgreSQL.

Each test asserts both what is returned and what must never be returned.
"""

from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path

import pytest

from rag.agent.tool_schemas import GetRelatedContextArgs
from rag.agent.tools import ToolExecutionError, get_related_context
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.pipeline import RetrievalPipeline


class _ConstantEmbedder:
    """Same unit vector for every text, so dense search returns the whole dataset."""

    def __init__(self, dimension: int) -> None:
        self._vector = [1.0] + [0.0] * (dimension - 1)

    def embed_query(self, text: str) -> list[float]:
        """Return the constant vector."""
        return self._vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return the constant vector per text."""
        return [self._vector for _ in texts]


class _Identity:
    """Reranker double that keeps order."""

    def rerank(self, query, results, top_n):
        """Return `results` truncated to `top_n`."""
        return results[:top_n]


def _write(root: Path, name: str, frontmatter: dict, body: str) -> Path:
    lines = ["---"]
    for key, value in frontmatter.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend(f'  - "{item}"' for item in value)
        else:
            lines.append(f'{key}: "{value}"')
    lines += ["---", "", body]
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


_LONG_BODY = "\n\n".join(f"Paragraph {i}. " + "retention wording " * 40 for i in range(6))


@pytest.fixture
def two_datasets(require_postgres, config):
    """Yield an ingestion pipeline and two disposable dataset ids, cleared afterwards."""
    embedder = _ConstantEmbedder(config.embedding.dimension)
    ingestion = IngestionPipeline(config, embedder=embedder, graph_client=None)
    a, b = f"pytest-scope-a-{uuid.uuid4()}", f"pytest-scope-b-{uuid.uuid4()}"
    try:
        yield ingestion, embedder, a, b
    finally:
        ingestion.clear_dataset(a)
        ingestion.clear_dataset(b)


def _pipeline(config, ingestion, embedder, *, authorization: bool) -> RetrievalPipeline:
    cfg = config.model_copy(deep=True)
    cfg.security.authorization.enabled = authorization
    cfg.retrieval.relationship_expansion.enabled = True
    return RetrievalPipeline(
        cfg, vectorstore=ingestion.vectorstore, embedder=embedder, reranker=_Identity()
    )


def _chunk_ids(ingestion, source: Path, dataset_id: str) -> list[str]:
    return [c.id for c in ingestion.vectorstore.get_chunks_by_source(str(source), dataset_id)]


# ---------------------------------------------------------------------------
# RET-013: chunk-id lookups are dataset- and tenant-scoped
# ---------------------------------------------------------------------------


def test_chunk_id_lookup_never_crosses_datasets(config, two_datasets, tmp_path):
    """A chunk id from dataset A resolves in A and nowhere else, by id or by section."""
    ingestion, embedder, a, b = two_datasets
    doc = _write(tmp_path / "a", "doc.md", {}, _LONG_BODY)
    ingestion.ingest_file(doc, a)
    ingestion.ingest_file(_write(tmp_path / "b", "other.md", {}, "Unrelated text."), b)
    store = ingestion.vectorstore
    ids = _chunk_ids(ingestion, doc, a)
    assert len(ids) > 1
    seed = store.get_chunks_by_ids([ids[0]], dataset_id=a)
    assert [c.id for c in seed] == [ids[0]]
    assert store.get_chunks_by_ids([ids[0]], dataset_id=b) == []
    assert store.get_chunks_by_ids([ids[0]], dataset_id=None) == []
    meta = seed[0].metadata
    assert store.get_chunks_by_section(meta.document_id, meta.section_path, dataset_id=b) == []
    assert store.get_chunks_by_section(meta.document_id, meta.section_path, dataset_id=a)


def test_get_related_context_refuses_another_datasets_chunk(config, two_datasets, tmp_path):
    """The agent/MCP tool returns nothing for another dataset's chunk id, and needs a dataset."""
    ingestion, embedder, a, b = two_datasets
    doc = _write(tmp_path / "a", "doc.md", {}, _LONG_BODY)
    ingestion.ingest_file(doc, a)
    pipeline = _pipeline(config, ingestion, embedder, authorization=False)
    seed_id = _chunk_ids(ingestion, doc, a)[0]
    args = GetRelatedContextArgs(chunk_id=seed_id)
    in_scope = get_related_context(args, pipeline, ingestion.vectorstore, None, a)
    assert in_scope and all(c.metadata.dataset_id == a for c in in_scope)
    assert get_related_context(args, pipeline, ingestion.vectorstore, None, b) == []
    with pytest.raises(ToolExecutionError):
        get_related_context(args, pipeline, ingestion.vectorstore, None, None)


def test_get_related_context_refuses_another_tenants_chunk(config, two_datasets, tmp_path):
    """Same dataset, other tenant: the seed and its neighbors are never returned."""
    ingestion, embedder, a, _ = two_datasets
    doc = _write(tmp_path, "ta.md", {"tenant_id": "ta", "allowed_roles": ["reader"]}, _LONG_BODY)
    ingestion.ingest_file(doc, a)
    pipeline = _pipeline(config, ingestion, embedder, authorization=True)
    args = GetRelatedContextArgs(chunk_id=_chunk_ids(ingestion, doc, a)[0])
    owner = AuthorizationContext(tenant_id="ta", roles=["reader"])
    intruder = AuthorizationContext(tenant_id="tb", roles=["reader"])
    assert get_related_context(args, pipeline, ingestion.vectorstore, owner, a)
    assert get_related_context(args, pipeline, ingestion.vectorstore, intruder, a) == []
    assert (
        ingestion.vectorstore.get_chunks_by_ids(
            [args.chunk_id], dataset_id=a, auth=pipeline.resolve_auth(intruder, {"dataset_id": a})
        )
        == []
    )


def test_relationship_expansion_looks_up_neighbors_in_the_results_own_dataset(
    config, two_datasets, tmp_path
):
    """Expansion passes each result's dataset to SQL; a mistagged result gets no neighbors."""
    from rag.schemas import SearchResult

    ingestion, embedder, a, b = two_datasets
    doc = _write(tmp_path / "a", "doc.md", {}, _LONG_BODY)
    ingestion.ingest_file(doc, a)
    pipeline = _pipeline(config, ingestion, embedder, authorization=False)
    seed = ingestion.vectorstore.get_chunks_by_ids(
        [_chunk_ids(ingestion, doc, a)[1]], dataset_id=a
    )[0]
    genuine = SearchResult(chunk=seed, score=1.0)
    mistagged = SearchResult(
        chunk=seed.model_copy(
            update={"metadata": seed.metadata.model_copy(update={"dataset_id": b})}
        ),
        score=1.0,
    )
    expanded = [r for r in pipeline.expand_with_relationships([genuine]) if r.origin == "expanded"]
    assert expanded and all(r.chunk.metadata.dataset_id == a for r in expanded)
    assert [
        r for r in pipeline.expand_with_relationships([mistagged]) if r.origin == "expanded"
    ] == []


# ---------------------------------------------------------------------------
# Lifecycle rules
# ---------------------------------------------------------------------------


def _names(pipeline, dataset_id, auth=None):
    results = pipeline.retrieve(
        "retention wording",
        filters={"dataset_id": dataset_id},
        candidate_k=50,
        generation_context_top_n=50,
        reranker_top_n=50,
        auth=auth,
    )
    return {Path(r.chunk.metadata.source).name for r in results}


def test_new_version_published_while_old_still_active(config, two_datasets, tmp_path):
    """v2 supersedes v1 but v1 still says active: only v2 is current; `as_of` still reaches v1."""
    ingestion, embedder, a, _ = two_datasets
    ingestion.ingest_file(
        _write(
            tmp_path, "p-v1.md", {"status": "active", "effective_from": "2025-01-01"}, _LONG_BODY
        ),
        a,
    )
    ingestion.ingest_file(
        _write(
            tmp_path,
            "p-v2.md",
            {"status": "active", "effective_from": "2026-01-01", "supersedes": "p-v1.md"},
            _LONG_BODY,
        ),
        a,
    )
    pipeline = _pipeline(config, ingestion, embedder, authorization=False)
    assert _names(pipeline, a) == {"p-v2.md"}
    assert _names(pipeline, a, AuthorizationContext(as_of=date(2025, 6, 1))) == {"p-v1.md"}


def test_branching_active_versions_keep_the_latest(config, two_datasets, tmp_path):
    """Two active successors of v1: the later `effective_from` is the one current version."""
    ingestion, embedder, a, _ = two_datasets
    ingestion.ingest_file(_write(tmp_path, "p-v1.md", {"status": "superseded"}, _LONG_BODY), a)
    for name, day in (("p-v2a.md", "2026-01-01"), ("p-v2b.md", "2026-03-01")):
        ingestion.ingest_file(
            _write(
                tmp_path,
                name,
                {"status": "active", "effective_from": day, "supersedes": "p-v1.md"},
                _LONG_BODY,
            ),
            a,
        )
    pipeline = _pipeline(config, ingestion, embedder, authorization=False)
    assert _names(pipeline, a) == {"p-v2b.md"}


def test_deleting_the_current_version_leaves_the_old_one_unlinked(config, two_datasets, tmp_path):
    """Rule: `status` alone hides nothing; once v2 is deleted, v1 has no family and is visible."""
    ingestion, embedder, a, _ = two_datasets
    ingestion.ingest_file(_write(tmp_path, "p-v1.md", {"status": "superseded"}, _LONG_BODY), a)
    v2 = _write(tmp_path, "p-v2.md", {"status": "active", "supersedes": "p-v1.md"}, _LONG_BODY)
    ingestion.ingest_file(v2, a)
    pipeline = _pipeline(config, ingestion, embedder, authorization=False)
    assert _names(pipeline, a) == {"p-v2.md"}
    assert ingestion.delete_sources(a, {str(v2)}) == 1
    assert _names(pipeline, a) == {"p-v1.md"}


# ---------------------------------------------------------------------------
# version_scope="history" (RET-014)
# ---------------------------------------------------------------------------


def _three_versions(ingestion, root: Path, dataset_id: str) -> None:
    for name, day, status, supersedes in (
        ("h-v1.md", "2025-01-01", "superseded", None),
        ("h-v2.md", "2026-01-01", "active", "h-v1.md"),
        ("h-v3.md", "2027-01-01", "draft", "h-v2.md"),
    ):
        meta = {"status": status, "effective_from": day}
        if supersedes:
            meta["supersedes"] = supersedes
        ingestion.ingest_file(_write(root, name, meta, _LONG_BODY), dataset_id)


def test_history_scope_returns_every_version_for_a_comparison(config, two_datasets, tmp_path):
    """Current scope keeps one version; history keeps all; history with a date drops later ones."""
    ingestion, embedder, a, _ = two_datasets
    _three_versions(ingestion, tmp_path, a)
    pipeline = _pipeline(config, ingestion, embedder, authorization=False)
    assert _names(pipeline, a) == {"h-v2.md"}
    history = AuthorizationContext(version_scope="history")
    assert _names(pipeline, a, history) == {"h-v1.md", "h-v2.md", "h-v3.md"}
    dated = AuthorizationContext(version_scope="history", as_of=date(2026, 6, 1))
    assert _names(pipeline, a, dated) == {"h-v1.md", "h-v2.md"}


def test_history_scope_never_bypasses_the_acl(config, two_datasets, tmp_path):
    """A reader of v1 but not v2 gets v1 only under history scope, never the restricted v2."""
    ingestion, embedder, a, _ = two_datasets
    ingestion.ingest_file(
        _write(
            tmp_path,
            "r-v1.md",
            {"tenant_id": "t1", "allowed_roles": ["reader", "admin"], "status": "superseded"},
            _LONG_BODY,
        ),
        a,
    )
    ingestion.ingest_file(
        _write(
            tmp_path,
            "r-v2.md",
            {
                "tenant_id": "t1",
                "allowed_roles": ["admin"],
                "status": "active",
                "supersedes": "r-v1.md",
            },
            _LONG_BODY,
        ),
        a,
    )
    pipeline = _pipeline(config, ingestion, embedder, authorization=True)
    reader = AuthorizationContext(tenant_id="t1", roles=["reader"], version_scope="history")
    other_tenant = AuthorizationContext(tenant_id="t2", roles=["admin"], version_scope="history")
    admin = AuthorizationContext(tenant_id="t1", roles=["admin"], version_scope="history")
    assert _names(pipeline, a, reader) == {"r-v1.md"}
    assert _names(pipeline, a, other_tenant) == set()
    assert _names(pipeline, a, admin) == {"r-v1.md", "r-v2.md"}
