# Freshness resolution benchmark (2026-11-22)

Script: `scripts/benchmark_freshness.py`. Raw results:
`experiments/results/freshness/*.json`. All data is synthetic: 30% of
documents in `supersedes` chains of 2 to 4 versions, 5 tenants, 10% null
tenant, 5% of base filenames reused across tenants. Setup (data generation,
inserts, `ANALYZE`) is timed separately and excluded from every figure
below. Medians unless stated; p95 is the nearest-rank 95th percentile of
5 to 30 repeats, so with 10 repeats it is close to the maximum.

## Conditions

| Run | Where | Code | Notes |
|---|---|---|---|
| `baseline_before` | Windows host, Python 3.14, 8 logical CPUs | before | Docker Desktop was starting; noisy |
| `after_indexed_families` | Windows host | indexed families | includes the frozen pairwise copy for comparison |
| `db_baseline_before_in_container` | `rag-api` container (2 CPUs, 2 GB), Python 3.11, PostgreSQL 16.14 + pgvector | before (git worktree of `cff0af8`) | 100 and 1,000 documents only (see below) |
| `db_after_in_container` | same container | indexed families, version fetch still `DISTINCT` | 100, 1,000, 10,000 documents, 3 chunks each |
| `db_host_indexed_families_artifact` | Windows host to the container's published port | indexed families | **not used for SQL conclusions**, see the artifact below |

The old-code container run was stopped at 10,000 documents: its timed
loops would have run the pairwise resolver (about 150 s per call) roughly
30 times. The 10,000-document "before" figure comes from one in-process
call instead.

## In-process family resolution

| Documents | Families | Indexed (current mode) | Indexed p95 | Indexed peak memory | Pairwise | Same result |
|---:|---:|---:|---:|---:|---:|---|
| 100 | 20 | 0.92 ms | 1.35 ms | 78 KiB | 10.5 ms | yes (both modes) |
| 1,000 | 190 | 11.1 ms | 12.5 ms | 764 KiB | 1,138 ms | yes (both modes) |
| 10,000 | 1,875 | 148 ms | 197 ms | 9.5 MiB | 151,606 ms (1 call) | yes (both modes) |
| 100,000 | 18,598 | 1,835 ms | 2,091 ms | 108 MiB | not run (projected 15,161 s) | n/a |

`after_indexed_families`, Windows host. `as_of` mode costs the same. The
pairwise resolver grows about 100x per 10x documents (quadratic); the
indexed one about 13x (linear plus Pydantic and set overhead).

## PostgreSQL, inside the compose network

### Version-metadata fetch (10,000 documents, 30,000 chunks)

| Query | Client median | Server execution | Rows identical |
|---|---:|---:|---|
| `SELECT DISTINCT ...` (previous) | 48.7 ms | 37.7 ms | reference |
| `SELECT DISTINCT ON (document_id) ...` | 51.2 ms | 34.7 ms | yes |
| `... AND chunk_index = 0` (now used) | 31.4 ms | 12.4 ms | yes |

`list_document_versions()` itself took 130 ms (p95 352 ms): building
10,000 `DocumentVersionInfo` objects costs more than the query.

### Dense search versus excluded-id array size (10,000 documents, `top_k=20`)

| Excluded ids | `store.search` | `NOT (id::text = ANY)` server | `id <> ALL(uuid[])` server |
|---:|---:|---:|---:|
| 0 | 22.0 ms | 20.0 ms | 18.7 ms |
| 1,000 | 26.4 ms | 26.4 ms | 18.8 ms |
| 3,705 (real exclusions) | 27.1 ms | 23.7 ms | 17.0 ms |
| 10,000 | 38.6 ms | 19.1 ms | 10.2 ms |

Every plan was an index scan on `dataset_id` plus a sort; pgvector's HNSW
index was never chosen for these filtered queries. Whether the `chunk_id`
tie-break in `ORDER BY` or the filter's selectivity is the reason was not
tested. As planned, exclusions cannot reduce recall, and dense search is
an exact scan of the dataset. The
array adds at most about 17 ms at 10,000 ids; the predicate was left
unchanged.

### Retrieval stages, medians (constant query vector, no reranker, no generation)

| Documents | Code | Freshness + ACL resolution | Dense search | BM25 | Total dense | Total hybrid |
|---:|---|---:|---:|---:|---:|---:|
| 100 | before | 8.4 ms | 3.4 ms | 6.0 ms | 12.2 ms | 19.2 ms |
| 100 | after | 4.0 ms | 4.9 ms | 7.4 ms | 9.4 ms | 15.6 ms |
| 1,000 | before | 891 ms | 7.2 ms | 28.1 ms | 898 ms | 937 ms |
| 1,000 | after | 30.9 ms | 10.2 ms | 32.8 ms | 42.3 ms | 60.5 ms |
| 10,000 | before | about 150 s (fetch plus one pairwise call) | n/a | n/a | n/a | n/a |
| 10,000 | after | 234 ms | 35.9 ms | 289 ms | 286 ms | 633 ms |

One 1,000-document dense sample took 4.0 s (p95 column of the raw file),
an isolated host stall; the hybrid run at the same size had p95 144 ms.
Embedding inference and generation were not measured; generation
dominates end-to-end latency on this host (EVAL-004) and is independent
of these changes.

## Measurement artifact: host to Docker Desktop

From the Windows host, any statement larger than about 16 KB took an extra
44 ms, whatever Postgres did (`SELECT 1` padded to 16.5 KB: 44.9 ms; to
16 KB: 0.42 ms). The same 19 KB statement from another container on the
compose network took 0.24 to 0.28 ms. `store.search` inlines the
384-float query vector twice, so adding an authorization clause or an
exclusion array pushed it over the threshold and made authorization look
about 10x slower (49 ms vs 5 ms) on the host. SQL conclusions above come
only from the in-container runs.

## Conclusions

- The quadratic family build was the dominant freshness cost (891 ms of a
  898 ms dense retrieval at 1,000 documents). Fixed.
- After the fix, at 10,000 documents per dataset, per-query freshness
  (about 230 to 320 ms) still exceeds dense search (36 ms). The cost is
  linear and mostly Python-side, which is what a maintained
  current-version projection would remove. At this project's corpus sizes
  (hundreds of documents) it is a few milliseconds.
- Exclusion arrays are not worth replacing at these sizes.
- 100,000 documents was measured in-process only; no database figure at
  that size exists.

## Scale run, 10,000 to 100,000 documents (2026-11-22, after the 37.1 changes)

Raw results: `scale_resolution_37_1_in_container.json`,
`scale_db_37_1_in_container.json`, `scale_db_fetch_psql_50k_100k.json`.
Same container (2 CPUs, 2 GB, Python 3.11), PostgreSQL container limited
to 1 GiB (`shared_buffers` 128MB, `work_mem` 4MB), 3 chunks per document,
HNSW index dropped before the inserts (no measured query uses it). The
resolver now applies the link-directed conflict rule and logs once per
resolution.

### In-process resolution and concurrency

| Documents | p50 | p95 | p99 | Peak memory | 4 threads p50 | 8 threads p50 / p99 | Throughput, 1 / 4 / 8 threads |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 134 ms | 184 ms | 184 ms | 7.2 MiB | 954 ms | 1.82 s / 2.96 s | 5.5 / 4.3 / 4.3 per s |
| 50,000 | 1.32 s | 2.01 s | 2.01 s | 43 MiB | 3.33 s | 6.62 s / 9.63 s | 0.55 / 1.18 / 1.13 per s |
| 100,000 | 2.25 s | 3.47 s | 3.47 s | 88 MiB | 8.07 s | 20.1 s / 26.7 s | 0.38 / 0.42 / 0.39 per s |

p95 and p99 come from 10 to 24 samples, so they are close to the maximum.
The previous pairwise resolver took 166,410 ms for one call at 10,000
documents in this container (families identical; the new conflict rule
adds 8 exclusions there), and projects to about 4,160 s at 50,000 and
16,640 s at 100,000, so it was not run at those sizes. Resolution is
CPU-bound Python under the GIL: extra threads do not raise throughput, they
only queue, so p50 grows roughly linearly with concurrency.

### PostgreSQL

At 10,000 documents (30,000 chunks): `list_document_versions` p50 117 ms,
the `chunk_index = 0` fetch 16.4 ms server-side against 95.0 ms for the old
`DISTINCT`; resolution 141 ms; dense retrieval total p50 292 ms (freshness
214 ms, dense search 45 ms); hybrid 849 ms (BM25 440 ms, **RET-006**).
Concurrent dense retrieval: p50 292 ms / 1.49 s / 2.99 s and p99 545 ms /
3.73 s / 4.87 s at 1 / 4 / 8 threads, 2.9 / 2.0 / 2.6 requests per second.
These samples overlapped a WAL checkpoint and an autovacuum of the freshly
loaded table (one fetch sample took 4.7 s), so treat them as noisy.

At 50,000 and 100,000 documents the version fetch alone took 22.6 to 41.3 s
with either query, run directly in `psql` (the table, about 1 GB with
vectors, does not fit in the container's memory, so every fetch reads it
from disk). The PostgreSQL server process exited three times during these
runs (`exited with exit code 2`, no preceding error, automatic recovery);
two of the three were during or right after the old `DISTINCT` query, and
the `chunk_index = 0` query never crashed in six runs. End-to-end and
concurrent retrieval at 50,000 and 100,000 were not measured at 1 GiB;
the 2 GiB rerun below has them.
The real `techfusion` data was intact after each recovery.

### Rerun at 2 GiB, and the crash investigation (2026-11-23)

The PostgreSQL crash was reproduced once more at 1 GiB, with the container's
cgroup counters sampled every 2 seconds (`crash_repro_memory_trace.log`):
memory pinned at exactly 1 GiB for the whole query phase, about 19,000
`memory.max` hits, ~65 MB of swap, and no OOM kill. Run alone four times
at 1 GiB, the old `DISTINCT` did not crash. The same full workload with the
limit raised to 2 GiB (`docker update`, reverted afterwards) completed with
no crash and no new limit hits, peaking at 1.72 GB. See **OPS-019** in
`KNOWN_ISSUES.md`.

That run (`scale_db_2g_37_1_in_container.json`, 3 chunks per document)
gives the end-to-end numbers missing above. At 50,000 documents the samples
overlapped the post-insert checkpoint and autovacuum and are not reliable.
At 100,000 documents:

| Measure | p50 | p95 |
|---|---:|---:|
| `chunk_index = 0` fetch (server) | 293 ms | |
| Old `DISTINCT` fetch (server) | 846 ms | |
| `list_document_versions()` (fetch plus objects) | 1.05 s | 2.95 s |
| Resolution on the fetched rows | 1.53 s | |
| Dense retrieval total | 3.11 s | 3.36 s |
| of which freshness | 2.50 s | |
| of which dense search | 630 ms | |
| Dense retrieval, 4 concurrent requests | 14.1 s | 19.5 s |

Throughput was 0.32 requests per second alone and 0.28 with 4 threads.
The 22 to 41 s fetch times above were measured at 1 GiB, where the table
did not fit in memory.

### What the scale run shows

Per-query freshness is fine at this project's corpus sizes and acceptable
at 10,000 documents per dataset, but at 50,000 and above it is not viable
on this host: the chunk-table fetch is I/O-bound and resolution is
CPU-bound and serialized: at 100,000 documents freshness is 2.5 s of a
3.1 s dense retrieval even with enough memory. This is the measured case for the maintained
projection (option 3 in `docs/topics/freshness-lifecycle.md`), or at least
a document-level version table with an index (option 2), before any
deployment with tens of thousands of documents per dataset.
