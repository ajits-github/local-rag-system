"""Benchmark document-version freshness resolution and its SQL predicates.

Part 1 always runs in-process over synthetic `DocumentVersionInfo` lists, compares
the current resolver with a frozen copy of the original pairwise one, and measures
concurrent resolution. Part 2 (`--db`) times the PostgreSQL fetch, exclusion
predicates, retrieval stages, and concurrent requests on disposable
`bench_fresh_*` tables, which are dropped afterwards.

Usage:
    python scripts/benchmark_freshness.py [--sizes 10000 50000 100000]
        [--db] [--db-sizes 10000 50000 100000] [--concurrency 1 4 8]
        [--out experiments/results/freshness/x.json]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
import tracemalloc
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.path_matching import source_matches_relevant  # noqa: E402
from rag.retrieval import freshness  # noqa: E402
from rag.retrieval.freshness import (  # noqa: E402
    resolve_current_document_source,
    resolve_excluded_document_ids,
)
from rag.schemas import DocumentVersionInfo  # noqa: E402

_AS_OF = date(2023, 6, 1)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def synthetic_versions(
    n: int,
    *,
    seed: int = 7,
    family_fraction: float = 0.3,
    max_chain: int = 4,
    tenants: int = 5,
    collision_fraction: float = 0.05,
) -> list[DocumentVersionInfo]:
    """Build `n` version records: standalone documents plus v1 -> vK chains.

    `supersedes_source` is a bare filename, as authored in front matter, so it
    is matched by path suffix. `collision_fraction` of the base names are
    reused under another tenant to exercise the cross-tenant link guard.
    """
    rng = random.Random(seed)
    versions: list[DocumentVersionInfo] = []
    seen_sources: set[str] = set()
    base_index = 0
    while len(versions) < n:
        tenant = None if rng.random() < 0.1 else f"tenant_{rng.randrange(tenants)}"
        category = f"cat_{rng.randrange(20)}"
        if base_index and rng.random() < collision_fraction:
            base = f"doc_{rng.randrange(base_index)}"
        else:
            base = f"doc_{base_index}"
        base_index += 1
        chain = rng.randint(2, max_chain) if rng.random() < family_fraction else 1
        chain = min(chain, n - len(versions))
        previous: str | None = None
        start = date(2020, 1, 1) + timedelta(days=rng.randrange(365))
        for k in range(1, chain + 1):
            filename = f"{base}-v{k}.md" if chain > 1 else f"{base}.md"
            source = f"data/{tenant or 'public'}/{category}/{filename}"
            if source in seen_sources:
                # (source, dataset_id) is unique in the real schema.
                source = f"data/{tenant or 'public'}/{category}/dup{base_index}/{filename}"
            seen_sources.add(source)
            versions.append(
                DocumentVersionInfo(
                    document_id=str(uuid.UUID(int=rng.getrandbits(128), version=4)),
                    source=source,
                    status="active" if k == chain else "superseded",
                    document_version=f"{k}.0",
                    effective_from=start + timedelta(days=200 * (k - 1)),
                    supersedes_source=previous,
                    tenant_id=tenant,
                )
            )
            previous = filename
    rng.shuffle(versions)
    return versions


# ---------------------------------------------------------------------------
# Frozen copy of the original pairwise resolver, for comparison only
# ---------------------------------------------------------------------------


def _legacy_links_to(candidate: DocumentVersionInfo, target: DocumentVersionInfo) -> bool:
    """Original `_links_to`: tenant guard plus path-suffix match."""
    if candidate.supersedes_source is None:
        return False
    if (
        candidate.tenant_id is not None
        and target.tenant_id is not None
        and candidate.tenant_id != target.tenant_id
    ):
        return False
    return source_matches_relevant(target.source, candidate.supersedes_source)


def _legacy_build_families(versions: list[DocumentVersionInfo]) -> list[list[DocumentVersionInfo]]:
    """Original O(N^2) union-find over every ordered pair."""
    parent = {v.document_id: v.document_id for v in versions}

    def find(doc_id: str) -> str:
        while parent[doc_id] != doc_id:
            parent[doc_id] = parent[parent[doc_id]]
            doc_id = parent[doc_id]
        return doc_id

    for v in versions:
        for other in versions:
            if v.document_id != other.document_id and _legacy_links_to(v, other):
                root_a, root_b = find(v.document_id), find(other.document_id)
                if root_a != root_b:
                    parent[root_a] = root_b
    groups: dict[str, list[DocumentVersionInfo]] = {}
    for v in versions:
        groups.setdefault(find(v.document_id), []).append(v)
    return [g for g in groups.values() if len(g) > 1]


def _legacy_excluded_for_current(family: list[DocumentVersionInfo]) -> set[str]:
    """Original current-mode rule: keep every active member."""
    active = {v.document_id for v in family if (v.status or "").strip().lower() == "active"}
    return set() if not active else {v.document_id for v in family} - active


def legacy_resolve_excluded(
    versions: list[DocumentVersionInfo], as_of: date | None, include_superseded: bool
) -> set[str]:
    """Original `resolve_excluded_document_ids` built on the pairwise family builder."""
    if as_of is None and include_superseded:
        return set()
    excluded: set[str] = set()
    for family in _legacy_build_families(versions):
        if as_of is not None:
            excluded |= freshness._excluded_for_as_of(family, as_of)
        else:
            excluded |= _legacy_excluded_for_current(family)
    return excluded


def _family_sets(families: list[list[DocumentVersionInfo]]) -> set[frozenset[str]]:
    """Families as comparable sets of document ids."""
    return {frozenset(v.document_id for v in family) for family in families}


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile of `values` (already in ms)."""
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered) + 0.5) - 1))
    return ordered[index]


def _stats(samples: list[float]) -> dict[str, float]:
    """p50/p95/p99/min/max of millisecond samples."""
    return {
        "n": len(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": _percentile(samples, 95),
        "p99_ms": _percentile(samples, 99),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def time_calls(fn: Callable[[], Any], repeats: int, warmup: int = 1) -> dict[str, float]:
    """Run `fn` `warmup` + `repeats` times; report percentile statistics in ms."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return _stats(samples)


def peak_memory_kib(fn: Callable[[], Any]) -> float:
    """Peak traced Python allocation during one call of `fn`, in KiB."""
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 1024


def peak_rss_mib() -> float | None:
    """Peak resident set size of this process so far, in MiB (Linux/macOS only)."""
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024 if sys.platform != "darwin" else peak / (1024 * 1024)


def concurrent_calls(fn: Callable[[], Any], threads: int, total: int) -> dict[str, Any]:
    """Run `total` calls of `fn` spread over `threads` workers; per-call latency and throughput."""
    samples: list[float] = []
    lock = threading.Lock()

    def one(_: int) -> None:
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000
        with lock:
            samples.append(elapsed)

    wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(one, range(total)))
    wall_s = time.perf_counter() - wall
    return {"threads": threads, **_stats(samples), "throughput_per_s": total / wall_s}


def _repeats_for(n: int) -> int:
    """More repeats for cheap sizes, fewer for expensive ones."""
    if n <= 1_000:
        return 30
    if n <= 10_000:
        return 20
    return 10


# ---------------------------------------------------------------------------
# Part 1: in-process resolution
# ---------------------------------------------------------------------------


def run_resolution_benchmark(
    sizes: list[int], max_legacy_seconds: float, concurrency: list[int]
) -> list[dict[str, Any]]:
    """Time current and legacy resolution at each size and compare their outputs."""
    rows = []
    legacy_last: tuple[int, float] | None = None
    for n in sizes:
        setup_start = time.perf_counter()
        versions = synthetic_versions(n)
        setup_ms = (time.perf_counter() - setup_start) * 1000
        families = freshness._build_families(versions)
        probe = next(v for v in versions if v.status == "superseded")
        repeats = _repeats_for(n)
        current = resolve_excluded_document_ids(versions, None, False)
        row: dict[str, Any] = {
            "n_documents": n,
            "n_families": len(families),
            "setup_ms_excluded_from_timings": setup_ms,
            "current": {
                "exclusions_current_mode": time_calls(
                    lambda v=versions: resolve_excluded_document_ids(v, None, False), repeats
                ),
                "exclusions_as_of_mode": time_calls(
                    lambda v=versions: resolve_excluded_document_ids(v, _AS_OF, False), repeats
                ),
                "resolve_current_document_source": time_calls(
                    lambda v=versions, p=probe: resolve_current_document_source(p.source, v),
                    repeats,
                ),
                "peak_memory_kib": peak_memory_kib(
                    lambda v=versions: resolve_excluded_document_ids(v, None, False)
                ),
                "excluded_count_current_mode": len(current),
                "concurrent": [
                    concurrent_calls(
                        lambda v=versions: resolve_excluded_document_ids(v, None, False),
                        threads,
                        max(threads * 3, 12),
                    )
                    for threads in concurrency
                ],
            },
        }
        projected = None
        if legacy_last is not None:
            prev_n, prev_ms = legacy_last
            projected = prev_ms * (n / prev_n) ** 2 / 1000
        if projected is not None and projected > max_legacy_seconds:
            row["legacy"] = {
                "skipped": True,
                "projected_seconds_per_call": projected,
                "reason": (
                    f"projected {projected:.0f}s per call from the measured O(N^2) growth "
                    f"exceeds --max-legacy-seconds={max_legacy_seconds:g}"
                ),
            }
        else:
            large = n >= 10_000
            captured: dict[str, Any] = {}

            def legacy_current(v=versions, c=captured) -> None:
                c["families"] = _legacy_build_families(v)

            legacy_timing = time_calls(
                legacy_current, 1 if large else repeats, warmup=0 if large else 1
            )
            legacy_last = (n, legacy_timing["p50_ms"])
            legacy_exclusions = set()
            for family in captured["families"]:
                legacy_exclusions |= _legacy_excluded_for_current(family)
            row["legacy"] = {
                "family_build_and_current_mode": legacy_timing,
                "families_identical": _family_sets(captured["families"]) == _family_sets(families),
                "as_of_exclusions_identical": legacy_resolve_excluded(versions, _AS_OF, False)
                == resolve_excluded_document_ids(versions, _AS_OF, False),
                "current_mode_extra_exclusions": len(current - legacy_exclusions),
                "current_mode_dropped_exclusions": len(legacy_exclusions - current),
            }
        rows.append(row)
        print(f"[resolution] n={n} done: {json.dumps(_summary(row))}", flush=True)
    return rows


def _summary(row: dict[str, Any]) -> dict[str, Any]:
    """Compact one-line progress summary for a resolution row."""
    legacy = row["legacy"]
    return {
        "current_p50_ms": round(row["current"]["exclusions_current_mode"]["p50_ms"], 3),
        "legacy_p50_ms": None
        if legacy.get("skipped")
        else round(legacy["family_build_and_current_mode"]["p50_ms"], 3),
    }


# ---------------------------------------------------------------------------
# Part 2: PostgreSQL
# ---------------------------------------------------------------------------

_BENCH_DOCUMENTS = "bench_fresh_documents"
_BENCH_CHUNKS = "bench_fresh_chunks"
_INSERT_BATCH_DOCS = 2_000


def run_db_benchmark(
    sizes: list[int],
    chunks_per_doc: int,
    keep: bool,
    keep_hnsw: bool,
    concurrency: list[int],
    hybrid_max: int,
    checkpoint: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Populate disposable tables per size, then time fetch, predicates, and retrieval stages.

    `checkpoint` receives the partial results after each size, so a crash
    keeps whatever finished.
    """
    import numpy as np
    import psycopg2
    import psycopg2.extras
    from pgvector.psycopg2 import register_vector

    from rag.config import load_config
    from rag.retrieval.authorization import AuthorizationContext
    from rag.retrieval.pipeline import RetrievalPipeline
    from rag.vectorstore.pgvector import PgVectorStore

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from init_db import build_schema_sql

    config = load_config()
    dsn = config.database_url()
    dimension = config.embedding.dimension
    rng = np.random.default_rng(11)

    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        server_version = cur.fetchone()[0]
        cur.execute(f"DROP TABLE IF EXISTS {_BENCH_CHUNKS}, {_BENCH_DOCUMENTS} CASCADE")
        cur.execute(
            build_schema_sql(
                documents_table=_BENCH_DOCUMENTS, chunks_table=_BENCH_CHUNKS, dimension=dimension
            )
        )
        if not keep_hnsw:
            # These filtered queries never choose the HNSW index (see the plans
            # recorded below), and maintaining it makes 100k-document inserts slow.
            cur.execute(f"DROP INDEX IF EXISTS {_BENCH_CHUNKS}_embedding_hnsw_idx")
    register_vector(conn)

    setup: dict[str, float] = {}
    versions_by_size: dict[int, list[DocumentVersionInfo]] = {}
    words = [f"term{i}" for i in range(2000)]
    now = datetime.now(UTC)
    try:
        for n in sizes:
            start = time.perf_counter()
            versions = synthetic_versions(n, seed=n)
            versions_by_size[n] = versions
            dataset_id = f"bench_{n}"
            for batch_start in range(0, n, _INSERT_BATCH_DOCS):
                batch = versions[batch_start : batch_start + _INSERT_BATCH_DOCS]
                doc_rows = [(v.document_id, v.source, dataset_id, "x" * 64) for v in batch]
                chunk_rows = []
                for v in batch:
                    vectors = rng.standard_normal((chunks_per_doc, dimension)).astype(np.float32)
                    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
                    for i in range(chunks_per_doc):
                        chunk_rows.append(
                            (
                                f"{v.document_id}_{i}",
                                v.document_id,
                                i,
                                " ".join(random.choices(words, k=60)),
                                vectors[i],
                                v.source,
                                "md",
                                dataset_id,
                                v.tenant_id,
                                v.status,
                                v.document_version,
                                v.effective_from,
                                v.supersedes_source,
                                now,
                                now,
                            )
                        )
                with conn.cursor() as cur:
                    psycopg2.extras.execute_values(
                        cur,
                        f"INSERT INTO {_BENCH_DOCUMENTS} "
                        "(document_id, source, dataset_id, checksum) VALUES %s",
                        doc_rows,
                        page_size=1000,
                    )
                    psycopg2.extras.execute_values(
                        cur,
                        f"""INSERT INTO {_BENCH_CHUNKS} (chunk_id, document_id, chunk_index,
                            content, embedding, source, source_type, dataset_id, tenant_id,
                            status, document_version, effective_from, supersedes_source,
                            created_at, last_modified) VALUES %s""",
                        chunk_rows,
                        page_size=500,
                    )
            setup[f"insert_{n}_ms"] = (time.perf_counter() - start) * 1000
            print(f"[db] inserted n={n}", flush=True)
        start = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(f"ANALYZE {_BENCH_DOCUMENTS}; ANALYZE {_BENCH_CHUNKS};")
        setup["analyze_ms"] = (time.perf_counter() - start) * 1000

        store = PgVectorStore(
            dsn,
            documents_table=_BENCH_DOCUMENTS,
            chunks_table=_BENCH_CHUNKS,
            maxconn=max(concurrency) + 1,
            statement_timeout_ms=600_000,
        )
        results: dict[str, Any] = {
            "server_version": server_version,
            "chunks_per_document": chunks_per_doc,
            "hnsw_index_kept": keep_hnsw,
            "setup_ms_excluded_from_timings": setup,
            "sizes": [],
        }
        for n in sizes:
            results["sizes"].append(
                _db_size_benchmark(
                    n,
                    conn,
                    store,
                    config,
                    rng,
                    dimension,
                    versions_by_size.pop(n),
                    AuthorizationContext,
                    RetrievalPipeline,
                    concurrency,
                    hybrid_max,
                )
            )
            print(f"[db] n={n} done", flush=True)
            checkpoint(results)
        return results
    finally:
        if not keep:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {_BENCH_CHUNKS}, {_BENCH_DOCUMENTS} CASCADE")
        conn.close()


_VERSION_FETCH_VARIANTS = {
    "distinct_all_columns": """SELECT DISTINCT document_id, source, status, document_version,
            effective_from, supersedes_source, tenant_id
        FROM {chunks} WHERE dataset_id = %s""",
    "first_chunk_only": """SELECT document_id, source, status, document_version,
            effective_from, supersedes_source, tenant_id
        FROM {chunks} WHERE dataset_id = %s AND chunk_index = 0""",
}


def _plan_summary(conn, sql: str, params: list[Any]) -> dict[str, Any]:
    """EXPLAIN ANALYZE `sql`; report the node types used and the server's own timing."""
    with conn.cursor() as cur:
        cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql, params)
        plan = cur.fetchone()[0][0]
    nodes: list[str] = []

    def walk(node: dict[str, Any]) -> None:
        label = node["Node Type"]
        if node.get("Index Name"):
            label += f" ({node['Index Name']})"
        nodes.append(label)
        for child in node.get("Plans", []):
            walk(child)

    walk(plan["Plan"])
    return {"nodes": nodes, "execution_ms": plan["Execution Time"]}


def _db_size_benchmark(
    n,
    conn,
    store,
    config,
    rng,
    dimension,
    versions,
    authorization_cls,
    pipeline_cls,
    concurrency,
    hybrid_max,
):
    """Time one dataset size's fetch, predicates, retrieval stages, and concurrent requests."""
    import numpy as np

    dataset_id = f"bench_{n}"
    repeats = 10 if n <= 10_000 else 5
    row: dict[str, Any] = {"n_documents": n}
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {_BENCH_CHUNKS} WHERE dataset_id = %s", (dataset_id,))
        row["n_chunks"] = cur.fetchone()[0]

    row["list_document_versions"] = time_calls(
        lambda: store.list_document_versions(dataset_id), repeats
    )
    variants = {}
    for name, sql in _VERSION_FETCH_VARIANTS.items():
        rendered = sql.format(chunks=_BENCH_CHUNKS)
        variants[name] = {
            **time_calls(lambda s=rendered: _raw(conn, s, [dataset_id]), repeats),
            "plan": _plan_summary(conn, rendered, [dataset_id]),
        }
    row["version_fetch_variants"] = variants

    fetched = store.list_document_versions(dataset_id)
    row["resolution_on_fetched_rows"] = time_calls(
        lambda f=fetched: resolve_excluded_document_ids(f, None, False), repeats
    )
    row["resolution_peak_memory_kib"] = peak_memory_kib(
        lambda f=fetched: resolve_excluded_document_ids(f, None, False)
    )
    excluded = sorted(resolve_excluded_document_ids(fetched, None, False))
    row["real_excluded_count"] = len(excluded)
    del fetched

    query_vector = rng.standard_normal(dimension).astype(np.float32)
    query_vector /= np.linalg.norm(query_vector)
    all_ids = [v.document_id for v in versions]
    search_rows = []
    for size in sorted({0, 1_000, 10_000, len(excluded)} & set(range(n + 1))):
        ids = excluded if size == len(excluded) else all_ids[:size]
        auth = authorization_cls(resolved_excluded_document_ids=ids, acl_enforced=False)
        search_rows.append(
            {
                "excluded_ids": size,
                "store_search": time_calls(
                    lambda a=auth: store.search(
                        query_vector.tolist(), top_k=20, filters={"dataset_id": dataset_id}, auth=a
                    ),
                    repeats,
                ),
            }
        )
    row["dense_search_by_exclusion_size"] = search_rows
    print(f"[db] n={n} fetch/search done {datetime.now(UTC).isoformat()}", flush=True)

    stage_config = config.model_copy(deep=True)
    stage_config.security.authorization.enabled = True
    stage_config.retrieval.relationship_expansion.enabled = False
    stage_config.reranker.provider = "none"
    stages = {}
    providers = ("dense", "hybrid") if n <= hybrid_max else ("dense",)
    acl_auth = authorization_cls(tenant_id="tenant_0", roles=["reader"])
    for provider in providers:
        stage_config.retrieval.provider = provider
        pipeline = pipeline_cls(
            stage_config, vectorstore=store, embedder=_FixedEmbedder(query_vector)
        )
        samples: dict[str, list[float]] = {}
        for _ in range(repeats):
            start = time.perf_counter()
            _, stage_ms = pipeline._retrieve_timed_inner(
                "term1 term2 term3", filters={"dataset_id": dataset_id}, auth=acl_auth
            )
            stage_ms["total_ms"] = (time.perf_counter() - start) * 1000
            for key, value in stage_ms.items():
                samples.setdefault(key, []).append(value)
        stages[provider] = {key: _stats(v) for key, v in samples.items()}
    row["retrieval_stages"] = stages
    if n > hybrid_max:
        row["hybrid_skipped_reason"] = (
            f"BM25 rebuilds an in-memory index over every chunk per query (RET-006); "
            f"not run above {hybrid_max} documents"
        )

    print(f"[db] n={n} stages done {datetime.now(UTC).isoformat()}", flush=True)
    stage_config.retrieval.provider = "dense"
    pipeline = pipeline_cls(stage_config, vectorstore=store, embedder=_FixedEmbedder(query_vector))
    concurrent = []
    for threads in concurrency:
        total = max(threads * 4, 12)
        print(f"[db] n={n} concurrency {threads} {datetime.now(UTC).isoformat()}", flush=True)
        concurrent.append(
            concurrent_calls(
                lambda p=pipeline: p.retrieve(
                    "term1", filters={"dataset_id": dataset_id}, auth=acl_auth
                ),
                threads,
                total,
            )
        )
    row["concurrent_dense_retrieve"] = concurrent
    row["process_peak_rss_mib"] = peak_rss_mib()
    return row


class _FixedEmbedder:
    """Returns one fixed query vector, so stage timings exclude model inference."""

    def __init__(self, vector) -> None:
        self._vector = vector.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Return the fixed vector."""
        return self._vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return the fixed vector per text."""
        return [self._vector for _ in texts]


def _raw(conn, sql: str, params: list[Any]):
    """Run `sql` and fetch all rows."""
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _machine_info() -> dict[str, Any]:
    """Host and revision details recorded with every result file."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--", "src"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        )
    except OSError:
        commit, dirty = "unknown", False
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "git_commit": commit,
        "src_dirty": dirty,
    }


def main() -> None:
    """Parse arguments, run the selected parts, and write one JSON report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 50_000, 100_000])
    parser.add_argument("--max-legacy-seconds", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--db", action="store_true", help="also run the PostgreSQL benchmark")
    parser.add_argument("--db-sizes", type=int, nargs="+", default=[10_000, 50_000, 100_000])
    parser.add_argument("--chunks-per-doc", type=int, default=3)
    parser.add_argument("--hybrid-max", type=int, default=10_000)
    parser.add_argument("--keep-hnsw", action="store_true", help="keep the HNSW index on insert")
    parser.add_argument("--keep", action="store_true", help="keep the bench_fresh_* tables")
    parser.add_argument("--skip-resolution", action="store_true", help="run only the --db part")
    parser.add_argument("--label", default="run")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out = Path(args.out or f"experiments/results/freshness/{args.label}.json")
    report: dict[str, Any] = {
        "label": args.label,
        "machine": _machine_info(),
        "synthetic_data": {
            "family_fraction": 0.3,
            "max_chain": 4,
            "tenants": 5,
            "null_tenant_fraction": 0.1,
            "collision_fraction": 0.05,
        },
    }
    if not args.skip_resolution:
        report["resolution"] = run_resolution_benchmark(
            sorted(args.sizes), args.max_legacy_seconds, args.concurrency
        )
        _write(out, report)
    if args.db:
        report["database"] = run_db_benchmark(
            sorted(args.db_sizes),
            args.chunks_per_doc,
            args.keep,
            args.keep_hnsw,
            args.concurrency,
            args.hybrid_max,
            lambda partial: _write(out, {**report, "database": partial}),
        )
    report["process_peak_rss_mib"] = peak_rss_mib()
    _write(out, report)
    print(f"wrote {out}")


def _write(out: Path, report: dict[str, Any]) -> None:
    """Write `report` as JSON to `out`."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
