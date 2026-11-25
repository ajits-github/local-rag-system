"""Freshness, authorization, and document lifecycle against a real PostgreSQL.

Every test ingests synthetic Markdown into its own disposable dataset with a
constant-vector embedder, so every chunk in the dataset is a candidate and
the assertions are about which documents the SQL predicates let through, on
both the dense and the BM25 path. Each security case asserts what is
returned and what must never be returned.
"""

from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path

import pytest

from rag.agent.tool_schemas import GetLatestDocumentArgs
from rag.agent.tools import get_latest_document
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


def _write(tmp_path: Path, name: str, frontmatter: dict, body: str) -> Path:
    """Write a Markdown file with a YAML front-matter block."""
    lines = ["---"]
    for key, value in frontmatter.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend(f'  - "{item}"' for item in value)
        else:
            lines.append(f'{key}: "{value}"')
    lines += ["---", "", body]
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture
def env(require_postgres, config, tmp_path):
    """Build an ingestion pipeline and dataset id; clears the dataset afterwards."""
    dataset_id = f"pytest-fresh-{uuid.uuid4()}"
    embedder = _ConstantEmbedder(config.embedding.dimension)
    ingestion = IngestionPipeline(config, embedder=embedder, graph_client=None)
    # Unrelated documents, so BM25 has a corpus in which query terms are rare.
    for i, words in enumerate(("network vlan", "backup nightly", "deploy canary", "badge door")):
        ingestion.ingest_file(_write(tmp_path / "filler", f"filler-{i}.md", {}, words), dataset_id)
    try:
        yield ingestion, embedder, dataset_id
    finally:
        ingestion.clear_dataset(dataset_id)


def _retrieval(config, ingestion, embedder, *, authorization, freshness_without_auth, hybrid):
    cfg = config.model_copy(deep=True)
    cfg.security.authorization.enabled = authorization
    cfg.retrieval.freshness.apply_without_authorization = freshness_without_auth
    cfg.retrieval.provider = "hybrid" if hybrid else "dense"
    cfg.retrieval.relationship_expansion.enabled = False
    return RetrievalPipeline(
        cfg, vectorstore=ingestion.vectorstore, embedder=embedder, reranker=_Identity()
    )


def _sources(results) -> set[str]:
    names = {Path(r.chunk.metadata.source).name for r in results}
    return {n for n in names if not n.startswith("filler-")}


def _retrieve(pipeline, dataset_id, auth=None, query="retention period days"):
    return _sources(
        pipeline.retrieve(
            query,
            filters={"dataset_id": dataset_id},
            candidate_k=50,
            generation_context_top_n=50,
            reranker_top_n=50,
            auth=auth,
        )
    )


def _keyword(pipeline, dataset_id, auth=None, query="retention period days"):
    resolved = pipeline.resolve_auth(auth, {"dataset_id": dataset_id})
    return _sources(
        pipeline.vectorstore.search_keyword(
            query, top_k=50, filters={"dataset_id": dataset_id}, auth=resolved
        )
    )


def _public_family(tmp_path: Path) -> list[Path]:
    v1 = _write(
        tmp_path,
        "retention-v1.md",
        {"status": "superseded", "effective_from": "2025-01-01", "document_version": "1"},
        "The retention period is 30 days.",
    )
    v2 = _write(
        tmp_path,
        "retention-v2.md",
        {
            "status": "active",
            "effective_from": "2026-01-01",
            "document_version": "2",
            "supersedes": "retention-v1.md",
        },
        "The retention period is 90 days.",
    )
    return [v1, v2]


@pytest.mark.parametrize("hybrid", [False, True])
def test_shipped_default_excludes_superseded_public_content(config, env, tmp_path, hybrid):
    """`config/default.yaml` as shipped: authorization off, only the active version retrievable."""
    from rag.config import load_config

    shipped = load_config()
    assert shipped.security.authorization.enabled is False
    assert shipped.retrieval.freshness.apply_without_authorization is True
    ingestion, embedder, ds = env
    for path in _public_family(tmp_path):
        ingestion.ingest_file(path, ds)
    cfg = shipped.model_copy(deep=True)
    cfg.retrieval.provider = "hybrid" if hybrid else "dense"
    cfg.retrieval.relationship_expansion.enabled = False
    pipeline = RetrievalPipeline(
        cfg, vectorstore=ingestion.vectorstore, embedder=embedder, reranker=_Identity()
    )
    assert _retrieve(pipeline, ds) == {"retention-v2.md"}


@pytest.mark.parametrize("hybrid", [False, True])
def test_compatibility_setting_still_returns_superseded_public_content(
    config, env, tmp_path, hybrid
):
    """`apply_without_authorization: false` keeps the earlier behavior: both versions visible."""
    ingestion, embedder, ds = env
    for path in _public_family(tmp_path):
        ingestion.ingest_file(path, ds)
    pipeline = _retrieval(
        config,
        ingestion,
        embedder,
        authorization=False,
        freshness_without_auth=False,
        hybrid=hybrid,
    )
    assert _retrieve(pipeline, ds) == {"retention-v1.md", "retention-v2.md"}


@pytest.mark.parametrize("authorization", [False, True])
def test_flag_excludes_superseded_public_content_in_sql(config, env, tmp_path, authorization):
    """Flag on: an anonymous caller gets only the active version, dense and BM25 alike."""
    ingestion, embedder, ds = env
    for path in _public_family(tmp_path):
        ingestion.ingest_file(path, ds)
    pipeline = _retrieval(
        config,
        ingestion,
        embedder,
        authorization=authorization,
        freshness_without_auth=True,
        hybrid=True,
    )
    assert _retrieve(pipeline, ds) == {"retention-v2.md"}
    assert _keyword(pipeline, ds) == {"retention-v2.md"}
    historical = AuthorizationContext(as_of=date(2025, 6, 1))
    assert _retrieve(pipeline, ds, historical) == {"retention-v1.md"}


def test_flag_never_hides_other_tenants_content_when_authorization_is_off(config, env, tmp_path):
    """Freshness-only filtering adds exclusions only; it never introduces tenant gating."""
    ingestion, embedder, ds = env
    for path in _public_family(tmp_path):
        ingestion.ingest_file(path, ds)
    ingestion.ingest_file(
        _write(tmp_path, "tenant-b.md", {"tenant_id": "tenant_b"}, "Retention period 7 days."), ds
    )
    pipeline = _retrieval(
        config, ingestion, embedder, authorization=False, freshness_without_auth=True, hybrid=False
    )
    assert _retrieve(pipeline, ds) == {"retention-v2.md", "tenant-b.md"}


def test_restricted_superseding_version_is_never_leaked_or_replaced_by_the_old_one(
    config, env, tmp_path
):
    """v2 is restricted: a v1-only reader sees neither version; a v2 reader sees only v2.

    Showing v1 as if current would present superseded content, so the family
    fails closed for a caller who cannot read its active member.
    """
    ingestion, embedder, ds = env
    v1 = _write(
        tmp_path,
        "limits-v1.md",
        {"tenant_id": "t1", "allowed_roles": ["reader", "admin"], "status": "superseded"},
        "The retention period is 30 days.",
    )
    v2 = _write(
        tmp_path,
        "limits-v2.md",
        {
            "tenant_id": "t1",
            "allowed_roles": ["admin"],
            "status": "active",
            "supersedes": "limits-v1.md",
        },
        "The retention period is 90 days.",
    )
    for path in (v1, v2):
        ingestion.ingest_file(path, ds)
    pipeline = _retrieval(
        config, ingestion, embedder, authorization=True, freshness_without_auth=False, hybrid=True
    )
    reader = AuthorizationContext(tenant_id="t1", roles=["reader"])
    admin = AuthorizationContext(tenant_id="t1", roles=["admin"])
    assert _retrieve(pipeline, ds, reader) == set()
    assert _keyword(pipeline, ds, reader) == set()
    assert _retrieve(pipeline, ds, admin) == {"limits-v2.md"}


def test_same_filename_in_two_tenants_never_forms_a_family(config, env, tmp_path):
    """Tenant B's `supersedes: pricing.md` must not exclude tenant A's `pricing.md`."""
    ingestion, embedder, ds = env
    a = _write(
        tmp_path, "a/pricing.md", {"tenant_id": "ta", "status": "active"}, "Retention period 1."
    )
    b_old = _write(
        tmp_path, "b/pricing.md", {"tenant_id": "tb", "status": "superseded"}, "Retention period 2."
    )
    b_new = _write(
        tmp_path,
        "b/pricing-2026.md",
        {"tenant_id": "tb", "status": "active", "supersedes": "pricing.md"},
        "Retention period 3.",
    )
    for path in (a, b_old, b_new):
        ingestion.ingest_file(path, ds)
    pipeline = _retrieval(
        config, ingestion, embedder, authorization=True, freshness_without_auth=False, hybrid=False
    )
    tenant_a = _retrieve(pipeline, ds, AuthorizationContext(tenant_id="ta"))
    assert tenant_a == {"pricing.md"}
    tenant_b = _retrieve(pipeline, ds, AuthorizationContext(tenant_id="tb"))
    assert tenant_b == {"pricing-2026.md"}


def test_missing_dataset_scope_keeps_tenant_isolation(config, env, tmp_path):
    """Without a dataset filter freshness is skipped, but another tenant is still invisible."""
    ingestion, embedder, ds = env
    ingestion.ingest_file(_write(tmp_path, "mine.md", {"tenant_id": "ta"}, "Retention zeta."), ds)
    ingestion.ingest_file(_write(tmp_path, "theirs.md", {"tenant_id": "tb"}, "Retention zeta."), ds)
    pipeline = _retrieval(
        config, ingestion, embedder, authorization=True, freshness_without_auth=True, hybrid=False
    )
    auth = pipeline.resolve_auth(AuthorizationContext(tenant_id="ta"), None)
    results = pipeline.vectorstore.search([1.0] + [0.0] * 383, top_k=500, auth=auth)
    names = {
        Path(r.chunk.metadata.source).name for r in results if r.chunk.metadata.dataset_id == ds
    }
    assert "mine.md" in names
    assert "theirs.md" not in names


def test_legacy_untagged_documents_stay_visible_to_every_caller(config, env, tmp_path):
    """No governance front matter means no tenant/role gate (documented convention)."""
    ingestion, embedder, ds = env
    ingestion.ingest_file(_write(tmp_path, "legacy.md", {}, "Retention period legacy."), ds)
    pipeline = _retrieval(
        config, ingestion, embedder, authorization=True, freshness_without_auth=False, hybrid=True
    )
    assert _retrieve(pipeline, ds, AuthorizationContext(tenant_id="anyone")) == {"legacy.md"}


def test_get_latest_document_follows_as_of_against_the_database(config, env, tmp_path):
    """The agent tool returns the version effective on `as_of`, not an empty result."""
    ingestion, embedder, ds = env
    for path in _public_family(tmp_path):
        ingestion.ingest_file(path, ds)
    pipeline = _retrieval(
        config, ingestion, embedder, authorization=True, freshness_without_auth=False, hybrid=False
    )
    chunks = get_latest_document(
        GetLatestDocumentArgs(source="retention-v2.md"),
        pipeline,
        ingestion.vectorstore,
        ds,
        "retention",
        embedder,
        AuthorizationContext(as_of=date(2025, 6, 1)),
        max_chunks=5,
        max_chunks_hard_ceiling=10,
    )
    assert {Path(c.metadata.source).name for c in chunks} == {"retention-v1.md"}


def test_content_update_replaces_chunks_under_a_stable_document_id(config, env, tmp_path):
    """Editing a file keeps its document_id; old text vanishes from dense and BM25."""
    ingestion, embedder, ds = env
    path = _write(tmp_path, "faq.md", {}, "The quokka limit is 30.")
    first = ingestion.ingest_file(path, ds)
    _write(tmp_path, "faq.md", {}, "The wombat limit is 45.")
    second = ingestion.ingest_file(path, ds)
    assert first["document_id"] == second["document_id"] and second["changed"]
    pipeline = _retrieval(
        config, ingestion, embedder, authorization=False, freshness_without_auth=False, hybrid=True
    )
    contents = [
        r.chunk.content
        for r in pipeline.retrieve(
            "quokka wombat",
            filters={"dataset_id": ds},
            candidate_k=50,
            generation_context_top_n=50,
            reranker_top_n=50,
        )
    ]
    assert any("wombat" in c for c in contents)
    assert not any("quokka" in c for c in contents)
    resolved = pipeline.resolve_auth(None, {"dataset_id": ds})
    keyword_hits = pipeline.vectorstore.search_keyword(
        "quokka", top_k=50, filters={"dataset_id": ds}, auth=resolved
    )
    # A chunk that no longer contains the term gets no keyword rank at all (RET-012).
    assert keyword_hits == []


def test_deleted_document_disappears_from_every_retrieval_path(config, env, tmp_path):
    """Hard delete: no dense, BM25, direct-fetch, or version-metadata trace remains."""
    ingestion, embedder, ds = env
    keep = _write(tmp_path, "keep.md", {}, "Retention period kept.")
    gone = _write(tmp_path, "gone.md", {}, "Retention period gone.")
    for path in (keep, gone):
        ingestion.ingest_file(path, ds)
    assert ingestion.delete_sources(ds, {str(gone)}) == 1
    pipeline = _retrieval(
        config, ingestion, embedder, authorization=False, freshness_without_auth=False, hybrid=True
    )
    store = ingestion.vectorstore
    assert _retrieve(pipeline, ds) == {"keep.md"}
    assert _keyword(pipeline, ds) == {"keep.md"}
    assert store.get_chunks_by_source(str(gone), ds) == []
    assert all(Path(v.source).name != "gone.md" for v in store.list_document_versions(ds))
    assert str(gone) not in store.list_document_sources(ds)


def test_version_fetch_returns_one_row_per_multi_chunk_document(config, env, tmp_path):
    """Reading only chunk 0 still yields every document exactly once, with its metadata."""
    ingestion, embedder, ds = env
    long_body = "\n\n".join(f"Paragraph {i}. " + "retention words " * 40 for i in range(12))
    for path in _public_family(tmp_path):
        ingestion.ingest_file(path, ds)
    big = _write(tmp_path, "big.md", {"status": "active"}, long_body)
    ingestion.ingest_file(big, ds)
    store = ingestion.vectorstore
    assert (
        store.count_chunks_by_document(ds)[
            store.get_or_create_document_id(str(big), "probe", ds)[0]
        ]
        > 1
    )
    versions = store.list_document_versions(ds)
    names = sorted(Path(v.source).name for v in versions)
    assert [n for n in names if not n.startswith("filler-")] == [
        "big.md",
        "retention-v1.md",
        "retention-v2.md",
    ]
    v2 = next(v for v in versions if v.source.endswith("retention-v2.md"))
    assert (v2.status, v2.supersedes_source, str(v2.effective_from)) == (
        "active",
        "retention-v1.md",
        "2026-01-01",
    )
