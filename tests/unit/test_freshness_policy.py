"""Freshness resolution, its decoupling from authorization, and the ACL invariants around it."""

from __future__ import annotations

import logging
import random
from datetime import UTC, date, datetime

import pytest

from rag.agent.tool_schemas import GetLatestDocumentArgs
from rag.agent.tools import get_latest_document
from rag.config import load_config
from rag.graph.authorization import (
    build_graph_authorization_predicate,
    build_graph_path_authorization_predicate,
    is_fact_authorized,
)
from rag.path_matching import source_matches_relevant
from rag.retrieval import freshness
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.freshness import resolve_current_document_source, resolve_excluded_document_ids
from rag.retrieval.pipeline import RetrievalPipeline
from rag.schemas import Chunk, ChunkMetadata, DocumentVersionInfo, SearchResult
from rag.vectorstore.pgvector import build_authorization_where_clause


def _v(doc_id, source, status=None, effective_from=None, supersedes_source=None, tenant_id=None):
    return DocumentVersionInfo(
        document_id=doc_id,
        source=source,
        status=status,
        effective_from=effective_from,
        supersedes_source=supersedes_source,
        tenant_id=tenant_id,
    )


_FAMILY = [
    _v("d1", "kb/policy-v1.md", status="superseded", effective_from=date(2025, 1, 1)),
    _v(
        "d2",
        "kb/policy-v2.md",
        status="active",
        effective_from=date(2026, 1, 1),
        supersedes_source="policy-v1.md",
    ),
]


# ---------------------------------------------------------------------------
# Indexed family construction
# ---------------------------------------------------------------------------


def _pairwise_families(versions):
    """Group documents with the original all-pairs union-find (reference)."""
    parent = {v.document_id: v.document_id for v in versions}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for v in versions:
        for other in versions:
            if v.document_id != other.document_id and freshness._links_to(v, other):
                parent[find(v.document_id)] = find(other.document_id)
    groups = {}
    for v in versions:
        groups.setdefault(find(v.document_id), set()).add(v.document_id)
    return {frozenset(g) for g in groups.values() if len(g) > 1}


@pytest.mark.parametrize("seed", range(25))
def test_indexed_families_match_the_pairwise_reference(seed):
    """Random corpora with nested paths, backslashes, null tenants and name collisions."""
    rng = random.Random(seed)
    names = [f"doc{i}.md" for i in range(12)]
    versions = []
    for i in range(60):
        tenant = rng.choice([None, "t1", "t2"])
        folder = rng.choice(["a", "a/b", "c", "a\\b"])
        versions.append(
            _v(
                f"id{i}",
                f"{folder}/{rng.choice(names)}" if rng.random() < 0.9 else rng.choice(names),
                supersedes_source=rng.choice(
                    [None, None, rng.choice(names), f"b/{rng.choice(names)}"]
                ),
                tenant_id=tenant,
            )
        )
    indexed = {frozenset(v.document_id for v in f) for f in freshness._build_families(versions)}
    assert indexed == _pairwise_families(versions)


def test_supersedes_link_never_crosses_tenants():
    """Same filename under two tenants: tenant B's v2 must not exclude tenant A's document."""
    versions = [
        _v("a1", "a/pricing.md", status="active", tenant_id="tenant_a"),
        _v("b1", "b/pricing-old.md", status="superseded", tenant_id="tenant_b"),
        _v(
            "b2",
            "b/pricing-new.md",
            status="active",
            supersedes_source="pricing.md",
            tenant_id="tenant_b",
        ),
    ]
    assert resolve_excluded_document_ids(versions, None, False) == set()


def test_suffix_match_requires_whole_path_segments():
    """`old-policy.md` is not a suffix match for `policy.md`."""
    assert not source_matches_relevant("kb/old-policy.md", "policy.md")
    versions = [
        _v("d1", "kb/old-policy.md", status="superseded"),
        _v("d2", "kb/policy-v2.md", status="active", supersedes_source="policy.md"),
    ]
    assert resolve_excluded_document_ids(versions, None, False) == set()


# ---------------------------------------------------------------------------
# Documented ambiguous-metadata semantics (pinned, not changed)
# ---------------------------------------------------------------------------


def test_no_active_member_keeps_every_version_retrievable():
    """No active member: nothing is excluded (documented, unchanged)."""
    versions = [
        _v("d1", "p-v1.md", status="draft"),
        _v("d2", "p-v2.md", status=None, supersedes_source="p-v1.md"),
    ]
    assert resolve_excluded_document_ids(versions, None, False) == set()


def test_status_comparison_is_case_and_whitespace_insensitive():
    """`status` is compared after strip/lower, so ` Active ` counts as active."""
    versions = [
        _v("d1", "p-v1.md", status="superseded"),
        _v("d2", "p-v2.md", status=" Active ", supersedes_source="p-v1.md"),
    ]
    assert resolve_excluded_document_ids(versions, None, False) == {"d1"}


def test_broken_supersedes_link_forms_no_family():
    """A `supersedes` target that matches no document links nothing."""
    versions = [_v("d2", "p-v2.md", status="active", supersedes_source="missing.md")]
    assert resolve_excluded_document_ids(versions, None, False) == set()


# ---------------------------------------------------------------------------
# resolve_current_document_source: deterministic matching
# ---------------------------------------------------------------------------


def test_exact_source_match_wins_over_a_longer_suffix_match():
    """An exact path match is chosen regardless of row order."""
    versions = [
        _v("d1", "policy.md", status="active"),
        _v("d2", "archive/policy.md", status="active"),
    ]
    for ordering in (versions, list(reversed(versions))):
        assert resolve_current_document_source("policy.md", ordering) == "policy.md"


def test_ambiguous_suffix_reference_is_not_resolved_by_row_order(caplog):
    """Two tenants' matches and no tenant hint: return the request unchanged, warn."""
    versions = [
        _v("a1", "a/policy-v1.md", status="superseded", tenant_id="tenant_a"),
        _v(
            "a2",
            "a/policy-v2.md",
            status="active",
            supersedes_source="a/policy-v1.md",
            tenant_id="tenant_a",
        ),
        _v("b1", "b/policy-v1.md", status="superseded", tenant_id="tenant_b"),
        _v(
            "b2",
            "b/policy-v2.md",
            status="active",
            supersedes_source="b/policy-v1.md",
            tenant_id="tenant_b",
        ),
    ]
    with caplog.at_level(logging.WARNING, logger="rag.retrieval.freshness"):
        for ordering in (versions, list(reversed(versions))):
            assert resolve_current_document_source("policy-v1.md", ordering) == "policy-v1.md"
    assert any("freshness_ambiguous_source_reference" in r.getMessage() for r in caplog.records)


def test_caller_tenant_disambiguates_a_shared_filename():
    """The caller's tenant picks its own family when a filename is shared."""
    versions = [
        _v("a1", "a/policy-v1.md", status="superseded", tenant_id="tenant_a"),
        _v(
            "a2",
            "a/policy-v2.md",
            status="active",
            supersedes_source="a/policy-v1.md",
            tenant_id="tenant_a",
        ),
        _v("b1", "b/policy-v1.md", status="superseded", tenant_id="tenant_b"),
        _v(
            "b2",
            "b/policy-v2.md",
            status="active",
            supersedes_source="b/policy-v1.md",
            tenant_id="tenant_b",
        ),
    ]
    for ordering in (versions, list(reversed(versions))):
        assert (
            resolve_current_document_source("policy-v1.md", ordering, tenant_id="tenant_a")
            == "a/policy-v2.md"
        )


# ---------------------------------------------------------------------------
# resolve_auth: freshness decoupled from authorization
# ---------------------------------------------------------------------------


class _Store:
    """Vector-store double recording calls and returning `_FAMILY` versions."""

    def __init__(self, chunks=None, excluded_by_authorization=0):
        self.version_calls = []
        self.search_calls = []
        self.chunks_by_source_calls = []
        self.count_calls = []
        self._chunks = chunks or []
        self._excluded = excluded_by_authorization

    def list_document_versions(self, dataset_id):
        self.version_calls.append(dataset_id)
        return list(_FAMILY)

    def search(self, query_embedding, top_k, filters=None, auth=None):
        self.search_calls.append(auth)
        return []

    def search_keyword(self, query, top_k, filters=None, auth=None):
        return []

    def count_excluded_by_authorization(self, filters, auth):
        self.count_calls.append(auth)
        return self._excluded

    def get_chunks_by_source(self, source, dataset_id, auth=None, limit=None):
        self.chunks_by_source_calls.append({"source": source, "auth": auth})
        return [c for c in self._chunks if c.metadata.source == source]


class _Embedder:
    def embed_query(self, text):
        return [0.0]

    def embed_documents(self, texts):
        return [[0.0] for _ in texts]


class _Reranker:
    def rerank(self, query, results, top_n):
        return results[:top_n]


def _pipeline(store, *, authorization: bool, freshness_without_auth: bool):
    config = load_config().model_copy(deep=True)
    config.security.authorization.enabled = authorization
    config.retrieval.freshness.apply_without_authorization = freshness_without_auth
    return RetrievalPipeline(config, vectorstore=store, embedder=_Embedder(), reranker=_Reranker())


def test_default_config_keeps_freshness_coupled_to_authorization():
    """Flag off and authorization off: unrestricted, no version fetch (historical behavior)."""
    store = _Store()
    pipeline = _pipeline(store, authorization=False, freshness_without_auth=False)
    caller = AuthorizationContext(tenant_id="tenant_a", roles=["reader"])
    assert pipeline.resolve_auth(caller, {"dataset_id": "ds"}) is None
    assert pipeline.resolve_auth(None, {"dataset_id": "ds"}) is None
    assert store.version_calls == []


@pytest.mark.parametrize("authorization", [False, True])
def test_freshness_only_context_for_an_unrestricted_caller(authorization):
    """An unrestricted caller gets exclusions only, with no tenant or roles."""
    store = _Store()
    pipeline = _pipeline(store, authorization=authorization, freshness_without_auth=True)
    resolved = pipeline.resolve_auth(None, {"dataset_id": "ds"})
    assert resolved is not None
    assert resolved.acl_enforced is False
    assert resolved.resolved_excluded_document_ids == ["d1"]
    assert resolved.tenant_id is None and resolved.roles == []


def test_authorization_off_freshness_only_context_drops_caller_acl_but_keeps_as_of():
    """Authorization off: caller ACL fields are dropped, `as_of` is kept."""
    store = _Store()
    pipeline = _pipeline(store, authorization=False, freshness_without_auth=True)
    caller = AuthorizationContext(
        tenant_id="tenant_a", roles=["reader"], as_of=date(2025, 6, 1), require_trust_level="high"
    )
    resolved = pipeline.resolve_auth(caller, {"dataset_id": "ds"})
    assert resolved.acl_enforced is False
    assert resolved.as_of == date(2025, 6, 1)
    assert resolved.resolved_excluded_document_ids == ["d2"]
    assert resolved.tenant_id is None and resolved.roles == []
    assert resolved.require_trust_level is None


def test_authorization_on_caller_context_is_always_acl_enforced():
    """A caller-built context can never switch ACL off once authorization is on."""
    store = _Store()
    pipeline = _pipeline(store, authorization=True, freshness_without_auth=True)
    forged = AuthorizationContext(tenant_id="tenant_a", roles=["reader"], acl_enforced=False)
    resolved = pipeline.resolve_auth(forged, {"dataset_id": "ds"})
    assert resolved.acl_enforced is True
    assert resolved.tenant_id == "tenant_a"
    assert resolved.resolved_excluded_document_ids == ["d1"]


@pytest.mark.parametrize("freshness_without_auth", [False, True])
def test_missing_dataset_scope_resolves_every_dataset_and_keeps_acl(freshness_without_auth, caplog):
    """No `dataset_id`: ACL still applies and freshness resolves across all datasets."""
    store = _Store()
    pipeline = _pipeline(store, authorization=True, freshness_without_auth=freshness_without_auth)
    caller = AuthorizationContext(tenant_id="tenant_a", roles=["reader"])
    with caplog.at_level(logging.INFO, logger="rag.retrieval.pipeline"):
        resolved = pipeline.resolve_auth(caller, None)
    assert resolved.tenant_id == "tenant_a" and resolved.acl_enforced is True
    assert resolved.resolved_excluded_document_ids == ["d1"]
    assert store.version_calls == [None]
    assert any("freshness_resolved_across_datasets" in r.getMessage() for r in caplog.records)


def test_resolves_freshness_mirrors_resolve_auth():
    """`resolves_freshness` is true exactly when `resolve_auth` fetches versions."""
    off = _pipeline(_Store(), authorization=False, freshness_without_auth=False)
    on = _pipeline(_Store(), authorization=False, freshness_without_auth=True)
    authz = _pipeline(_Store(), authorization=True, freshness_without_auth=False)
    caller = AuthorizationContext(tenant_id="t")
    assert not off.resolves_freshness(caller)
    assert on.resolves_freshness(None)
    assert authz.resolves_freshness(caller) and not authz.resolves_freshness(None)


def test_retrieve_passes_freshness_only_context_to_dense_and_keyword_search():
    """Hybrid retrieval passes the freshness-only context to the store."""
    store = _Store()
    config_pipeline = _pipeline(store, authorization=False, freshness_without_auth=True)
    config_pipeline._config.retrieval.provider = "hybrid"
    config_pipeline.retrieve("q", filters={"dataset_id": "ds"})
    (passed,) = store.search_calls
    assert passed.acl_enforced is False
    assert passed.resolved_excluded_document_ids == ["d1"]


def test_freshness_only_context_never_emits_authorization_denied():
    """Thin results under freshness-only filtering are not an ACL denial."""
    store = _Store(excluded_by_authorization=5)
    pipeline = _pipeline(store, authorization=False, freshness_without_auth=True)
    pipeline.retrieve("q", filters={"dataset_id": "ds"})
    assert store.count_calls == []


# ---------------------------------------------------------------------------
# SQL and graph predicates
# ---------------------------------------------------------------------------


def test_freshness_only_sql_has_exclusions_and_no_acl_predicate():
    """A freshness-only context yields only the exclusion predicate."""
    auth = AuthorizationContext(resolved_excluded_document_ids=["d1"], acl_enforced=False)
    sql, params = build_authorization_where_clause(auth, ["support"])
    assert sql == "NOT (document_id::text = ANY(%s))"
    assert params == [["d1"]]


def test_freshness_only_sql_with_nothing_excluded_is_unrestricted():
    """A freshness-only context with no exclusions adds no SQL at all."""
    sql, params = build_authorization_where_clause(AuthorizationContext(acl_enforced=False), [])
    assert (sql, params) == ("", [])


def test_acl_sql_is_unchanged_by_the_marker():
    """An ACL-enforcing context still produces every tenant/role/trust term."""
    auth = AuthorizationContext(
        tenant_id="t",
        roles=["r"],
        resolved_excluded_document_ids=["d1"],
        require_trust_level="high",
    )
    sql, params = build_authorization_where_clause(auth, ["support"])
    assert "tenant_id = %s" in sql and "allowed_roles && %s::text[]" in sql
    assert "NOT (document_id::text = ANY(%s))" in sql and "trust_level = %s" in sql
    assert params == ["t", [], ["r"], ["d1"], "high"]


def test_graph_freshness_only_predicate_has_no_acl_terms():
    """Cypher predicates for a freshness-only context carry only exclusions."""
    auth = AuthorizationContext(resolved_excluded_document_ids=["d1"], acl_enforced=False)
    clause, params = build_graph_authorization_predicate(
        auth, node_var="e", cross_tenant_support_roles=[]
    )
    assert "tenant_id" not in clause and "allowed_roles" not in clause
    assert params == {"_auth_excluded_document_ids": ["d1"]}
    empty = AuthorizationContext(acl_enforced=False)
    assert (
        build_graph_path_authorization_predicate(
            empty, path_var="path", cross_tenant_support_roles=[]
        )
        == ""
    )


def test_graph_python_recheck_for_freshness_only_context():
    """The Python graph re-check applies exclusions only, and ACL stays for ACL contexts."""
    auth = AuthorizationContext(resolved_excluded_document_ids=["d1"], acl_enforced=False)
    other_tenant = {"tenant_id": "tenant_b", "allowed_roles": ["x"], "source_document_id": "d9"}
    assert is_fact_authorized(other_tenant, auth, cross_tenant_support_roles=[])
    assert not is_fact_authorized({"source_document_id": "d1"}, auth, cross_tenant_support_roles=[])
    acl = AuthorizationContext(tenant_id="tenant_a", roles=[])
    assert not is_fact_authorized(other_tenant, acl, cross_tenant_support_roles=[])


# ---------------------------------------------------------------------------
# get_latest_document honors as_of
# ---------------------------------------------------------------------------


def _chunk(doc_id, source):
    now = datetime.now(UTC)
    return Chunk(
        id=f"{doc_id}_0",
        content=f"content of {source}",
        metadata=ChunkMetadata(
            document_id=doc_id,
            chunk_id=f"{doc_id}_0",
            source=source,
            source_type="md",
            created_at=now,
            last_modified=now,
            chunk_index=0,
            dataset_id="ds",
        ),
    )


def test_get_latest_document_resolves_to_the_version_effective_on_as_of():
    """Historical query: the redirect agrees with the `as_of` exclusions."""
    store = _Store(chunks=[_chunk("d1", "kb/policy-v1.md"), _chunk("d2", "kb/policy-v2.md")])
    pipeline = _pipeline(store, authorization=True, freshness_without_auth=False)
    caller = AuthorizationContext(tenant_id="t", roles=["r"], as_of=date(2025, 6, 1))
    chunks = get_latest_document(
        GetLatestDocumentArgs(source="policy-v2.md"),
        pipeline,
        store,
        "ds",
        "query",
        _Embedder(),
        caller,
        max_chunks=5,
        max_chunks_hard_ceiling=10,
    )
    assert [c.metadata.source for c in chunks] == ["kb/policy-v1.md"]
    (call,) = store.chunks_by_source_calls
    assert call["auth"].resolved_excluded_document_ids == ["d2"]


def test_get_latest_document_current_mode_still_redirects_to_active():
    """Without `as_of`, an old path still redirects to the active version."""
    store = _Store(chunks=[_chunk("d1", "kb/policy-v1.md"), _chunk("d2", "kb/policy-v2.md")])
    pipeline = _pipeline(store, authorization=True, freshness_without_auth=False)
    chunks = get_latest_document(
        GetLatestDocumentArgs(source="policy-v1.md"),
        pipeline,
        store,
        "ds",
        "query",
        _Embedder(),
        AuthorizationContext(tenant_id="t", roles=["r"]),
        max_chunks=5,
        max_chunks_hard_ceiling=10,
    )
    assert [c.metadata.source for c in chunks] == ["kb/policy-v2.md"]
    assert store.version_calls == ["ds"]


def test_search_result_type_is_unaffected():
    """Sanity check that SearchResult still accepts these chunks (no schema change)."""
    assert SearchResult(chunk=_chunk("d1", "a.md"), score=1.0).origin == "retrieved"


def test_history_scope_reaches_the_context_and_is_audited(caplog):
    """`version_scope` survives resolution in both ACL modes and is audit-logged."""
    for authorization in (False, True):
        store = _Store()
        pipeline = _pipeline(store, authorization=authorization, freshness_without_auth=True)
        caller = AuthorizationContext(tenant_id="t", roles=["r"], version_scope="history")
        with caplog.at_level(logging.INFO, logger="rag.audit"):
            resolved = pipeline.resolve_auth(caller, {"dataset_id": "ds"})
        assert resolved.version_scope == "history"
        assert resolved.resolved_excluded_document_ids == []
        assert resolved.acl_enforced is authorization
        if authorization:
            assert (resolved.tenant_id, resolved.roles) == ("t", ["r"])
    assert any("freshness_version_history_requested" in r.getMessage() for r in caplog.records)


def test_source_label_shows_version_and_status():
    """The prompt can tell an old version from the current one."""
    from rag.retrieval.pipeline import source_label

    chunk = _chunk("d1", "kb/policy-v1.md")
    chunk.metadata.document_version = "1.0"
    chunk.metadata.status = "superseded"
    label = source_label(SearchResult(chunk=chunk, score=1.0))
    assert "version 1.0 (superseded)" in label
    assert "version" not in source_label(SearchResult(chunk=_chunk("d9", "kb/plain.md"), score=1.0))
