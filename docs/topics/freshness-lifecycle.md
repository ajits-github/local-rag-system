# Freshness & Document Lifecycle

How the system decides which version of a document is current, where that
decision is enforced, what happens to a document when its file changes or
disappears, and how a local directory can be kept in sync on a schedule.
Benchmarks behind the decisions here are in
[`experiments/reports/freshness_benchmark.md`](https://github.com/ajits-github/local-rag-system/blob/main/experiments/reports/freshness_benchmark.md).

## Version families

A document joins a version family through its `supersedes` front-matter
field, stored as `supersedes_source` and matched against other documents'
`source` paths by whole-segment path suffix
(`rag.path_matching.source_matches_relevant`). Two documents with
different non-null `tenant_id`s, or in different datasets, never link.
Families are built per query from `VectorStore.list_document_versions`
with a path-suffix index, so the cost is linear in the number of documents.

| Situation | Current query (`as_of` unset) | Historical query (`as_of` set) |
|---|---|---|
| One `status: active` member | Every other member excluded | Member with the latest `effective_from <= as_of` kept |
| Several active members, one supersedes the others (directly or transitively) | Only the superseding one kept; `freshness_multiple_active_family_members` logged | as above |
| Several active members the links cannot order (branches, cycles) | Latest `effective_from` kept (missing dates sort first), then greatest `document_id`; `freshness_conflicting_active_versions` logged | as above |
| No active member | Nothing excluded | as above |
| Member without `effective_from` | (status decides) | Never excluded |
| `as_of` before every dated member | (n/a) | Every dated member excluded |
| Tie on `effective_from` | (see above) | Lexicographically greatest `document_id` wins |
| `supersedes` names nothing | No family | No family |
| `include_superseded=True` | Nothing excluded | Ignored |
| `version_scope: history` | Nothing excluded | Members effective after `as_of` excluded; every earlier or undated member kept |

`status` is compared case- and whitespace-insensitively; `status` alone
never hides a document that is not in a family. Each resolution logs at
most one warning per kind, with a count and sample ids. The rules are
pinned by `tests/unit/test_freshness.py`,
`tests/unit/test_freshness_lifecycle_rules.py`, and
`tests/unit/test_freshness_policy.py`.

### Comparing versions

A question about what changed between versions needs the superseded text
too. Requests (`POST /query`, the agent routes, `/graph/query`, and the MCP
RAG tools) take `version_scope`: `"current"` (default) applies the table's
current or `as_of` column; `"history"` applies its last row, returning
every version already effective on `as_of`. It is a version selector, not
an authorization switch: tenant, role, and trust predicates apply
unchanged, and each use logs a `freshness_version_history_requested` audit
event. Evidence labels carry `version X (status)` in either scope, so the
prompt shows which text is superseded. `include_superseded` remains an
internal field that no request sets.

## Where freshness and authorization are enforced

All filtering below happens in the database query (SQL predicate) or in
the Cypher query plus a Python re-check, never in answer formatting.

| Path | Resolution | Enforced in |
|---|---|---|
| `retrieve()`, dense and hybrid/BM25 | `resolve_auth` | `search`/`search_keyword` SQL |
| `retrieve_attribution()` | `resolve_auth` | same SQL |
| Relationship expansion | resolved context from `retrieve()` | `get_chunks_by_ids`/`get_chunks_by_section` SQL, scoped to each result's own `dataset_id` |
| Agent `search_knowledge_base` | via `retrieve()` | same SQL |
| Agent `get_document` | `resolve_auth` | `get_chunks_by_source` SQL (a superseded path returns nothing) |
| Agent `get_latest_document` | `resolve_auth`, then redirect with the resolved `as_of` | `get_chunks_by_source` SQL |
| Agent `get_related_context` (requires `dataset_id`) | `resolve_auth` | `get_chunks_by_ids` SQL, scoped to `dataset_id` |
| Graph retrieval and graph tools | `resolve_auth` | Cypher predicate, Python re-check, then Postgres revalidation (`get_document_governance`) |
| MCP RAG tools | same tool functions, after `retrieval.dataset_scope` | same as above |

After retrieval, `sanitize_evidence` only redacts and flags; Context
Engineering only reorders by freshness tier. Neither filters documents.

### Scope and the authorization switch

Freshness applies to every query by default
(`retrieval.freshness.apply_without_authorization: true`). When
authorization is off, or the request carries no authorization context,
`resolve_auth` returns a freshness-only context (`acl_enforced=False`). It
carries only the caller's `as_of`/`include_superseded` and the resolved
exclusions; the SQL and Cypher builders emit no tenant, role, or trust
term for it, so it can only remove superseded versions, never hide or
reveal anything else. With authorization enabled, a caller's context is
always `acl_enforced=True`, whatever context it arrived with.

`apply_without_authorization: false` restores the earlier behavior, where
only authorized requests resolved freshness. Experiment configs recorded
before 2026-11-22 ran with that behavior; set the flag to `false` to
reproduce them exactly.

### Requests without a `dataset_id`

`retrieval.dataset_scope` decides, at the API routers (`/query`,
`/agent/query`, `/agent/query/stream`, `/graph/query`) and in the MCP
server, via `rag.retrieval.scope`:

| Deployment | Setting | A request without `dataset_id` |
|---|---|---|
| Single dataset | `default_dataset_id: <id>` | Scoped to that dataset |
| Several datasets | `missing_dataset_policy: all_datasets` (default) | Searches every dataset; freshness resolves each dataset's families separately (`freshness_resolved_across_datasets`) |
| Several datasets | `missing_dataset_policy: reject` | HTTP 422, or an MCP tool error |

Freshness is therefore never skipped for lack of a dataset, and tenant and
role enforcement never depend on dataset scope. Unscoped resolution reads
every dataset's version rows, so its cost grows with the whole corpus.
`get_document`, `get_latest_document`, `get_related_context`, and the graph
tools still require a dataset of their own. Chunk-id lookups
(`get_chunks_by_ids`, `get_chunks_by_section`) always filter by
`dataset_id` and the authorization predicate in SQL.

### A restricted current version

If a caller may read v1 but not its active successor v2, the caller sees
neither: v1 is excluded as superseded and v2 by its ACL. Returning v1
would present stale content as current. Proven against PostgreSQL in
`test_restricted_superseding_version_is_never_leaked_or_replaced_by_the_old_one`.

## Document lifecycle

| Event | What happens | Identity |
|---|---|---|
| **Content update** (same file edited) | `replace_document_chunks` commits the new checksum, deletes the old chunks, and inserts the new ones in one transaction | `document_id` unchanged; `chunk_id`s reused by position |
| **Published new version** (new file with `supersedes`) | Ingested as its own document; becomes current for queries once its transaction commits | New `document_id`; the old version stays stored for `as_of` queries |
| **Deletion** (file gone from a scanned root) | Hard delete of the `documents` row; chunks cascade; graph facts deleted by document | Gone; a re-added file gets a new `document_id` |

There is no tombstone or retention period: a deleted document cannot be
queried historically. Publishing a new version is atomic per file, not
across files: if the new version's ingestion fails, the old one simply
stays current. A new version that supersedes an old one still marked
`status: active` is current as soon as it commits; the old one is
excluded by the link and a warning is logged. Deleting the current
version dissolves the family, so its predecessor becomes visible again
(its `status` alone does not hide it). Each rule is tested against
PostgreSQL in `tests/integration/test_chunk_scope_and_lifecycle.py` and
`tests/integration/test_freshness_lifecycle.py`.

No answer or evidence cache exists. The only caches are a token-count
cache keyed by text hash (`context/caching.py`), the image-description
cache keyed by image checksum, and RAGAS evaluation caches, so no
invalidation step is needed after an update or delete.

Linking by an explicit `supersedes_document_id` or a stable family id
would be more reliable than filename suffixes, but `document_id` is
assigned at ingestion and unknown when front matter is authored. A
family id would need a schema column, a backfill from today's links, and
a front-matter field; that is deferred to the decision below.

## Scheduled directory ingestion

`python -m rag.ingestion.scheduler [--once]` keeps configured directories
in sync with their datasets. It refuses to start unless
`ingestion.scheduled_scan.enabled` is true.

```yaml
ingestion:
  scheduled_scan:
    enabled: true
    interval_seconds: 3600
    state_dir: data/ingestion_state
    settle_seconds: 30
    max_delete_fraction: 0.5
    sources:
      - {path: data/knowledge_base, dataset_id: techfusion, tenant_id: null}
```

Each round, per source:

1. Takes an in-process lock and a lock file in `state_dir`. A second
   trigger reports `locked` and does nothing; a lock file older than
   `lock_stale_seconds` is taken over, and a running scan refreshes it.
2. Checks the directory exists and is readable, else `root_unavailable`.
3. Walks it (hidden entries, `~$`/`.#` lock files and `*.tmp`/`*.swp`-style
   temporaries skipped; extensions matched case-insensitively). Files
   modified within `settle_seconds` are deferred.
4. Skips a file whose stored checksum equals its sha256 without loading it,
   when no vision provider or graph projection makes the checksum differ
   from the plain file digest; otherwise calls `ingest_file`. The manifest
   hint (size, mtime, inode) lets an untouched file skip hashing; an atomic
   write-then-rename replacement changes the inode and is re-hashed even if
   size and mtime match. A file whose path was stored under another
   spelling of the same root keeps that spelling, so its `document_id` is
   stable.
5. Deletes documents whose files are gone, only if the walk had no errors,
   at least one file was found, and the round would not remove more than
   `max_delete_fraction` of the source's documents. Each deletion is
   re-checked against the filesystem first.
6. Writes a manifest (last report, per-file size/mtime/sha256 hints)
   atomically, and records `rag_ingestion_scan_*` metrics and an
   `ingestion_scan_completed` log line.

The database, not the manifest, decides what changed, so losing the
manifest only costs re-hashing. A file that fails to ingest is reported
(`partial`) and never deleted. Renames are counted as `probable_renames`
but processed as delete plus add, since a moved file cannot be told apart
from a removed one plus an unrelated new one. Locking is per host; two
hosts scanning one share would need a database advisory lock.

`IngestionPipeline.ingest_path` (the CLI and `make ingest`) shares the
same walk (`discover_files`), spelling reuse, and deletion guard
(`deletion_blocked_reason`), with `ingestion.max_delete_fraction`. When it
withholds deletion it reports `deletion_blocked_reason` and
`deletion_candidates` in `IngestionStats`, and the CLI exits with code 3.
To delete intentionally, pass `--allow-mass-delete`
(`allow_mass_delete=True`), which overrides the empty-directory and
fraction guards but never an incomplete walk; `--clear` still empties a
whole dataset first.

## Source registry and connectors (design only)

Not implemented. A future SharePoint, Confluence, or Git source would
register a row per source:

| Field | Purpose |
|---|---|
| `source_id` | Stable id; namespaces document identity |
| `provider` | `local`, `git`, `sharepoint`, `confluence` |
| `tenant_id`, `dataset_id` | Where documents land and who owns them |
| `location` | Path, repository URL, site, or space |
| `sync_cursor` | Provider checkpoint (commit SHA, delta token, `lastModified`) |
| `last_success_at`, `status`, `last_error` | Operations |
| `deletion_policy` | `hard_delete`, `retain_days`, or `never` |

A connector would implement `list_changes(cursor) -> (changes, new_cursor)`
where each change is `upsert(external_id, bytes, metadata)` or
`delete(external_id)`, with `source = f"{source_id}:{external_id}"` so
identity survives renames on providers that have stable ids, and
`ingest_file(..., source_override=...)` doing the work. The cursor
advances only after a batch commits. Permission sync (mapping provider
ACLs to `allowed_roles`), OAuth, webhooks, and API quotas are deferred
until a real source needs them; the local scheduler above is the only
implementation today.

## Decision: searchable-state projection

**Status:** deferred, not implemented. Freshness now runs on every query
by default, and the scale run shows per-query resolution is not viable at
50,000 documents per dataset on this host, so the projection (or at least
option 2) is required before any deployment of that size.

Options compared:

1. **Per-query resolution with indexed family matching** (current).
   Correct by construction, no schema change, nothing to keep consistent
   after a crash. Cost per query grows linearly with the dataset's
   document count. At 10,000 documents the whole step (fetch plus
   resolution) took 214 to 234 ms against 36 to 45 ms for dense search,
   and the exclusion array added at most 17 ms at 10,000 ids. At 50,000
   and 100,000 documents resolution alone took 1.3 s and 2.3 s (20 s p50
   with 8 concurrent requests, since it is CPU-bound under the GIL). With
   2 GiB for PostgreSQL, a dense retrieval at 100,000 documents took
   3.1 s p50, 2.5 s of it freshness; with 1 GiB the chunk-table fetch
   took 22 to 41 s because the table no longer fit in memory.
2. **Document-level version table** (governance columns plus an index on
   `documents`). Removes the chunk-level fetch, but resolution and the
   exclusion array stay per query.
3. **Transactionally maintained projection** (`is_current` flag and
   `[effective_from, effective_to)` range per document, updated in the
   same transaction as `replace_document_chunks` and deletions). Current
   queries become `WHERE is_current` with no fetch and no array; `as_of`
   queries a range predicate. Costs: a migration with backfill, family
   recomputation on every write (a new version changes its predecessor's
   row), tenant-scoped recomputation, and a dual-read comparison period
   behind a flag, with rollback by switching the flag off.

Option 3 is the right end state at scale, and option 2 is a cheap first
step, but at the corpus sizes this project runs (hundreds of documents)
per-query resolution measured 4 ms at 100 documents and 31 ms at 1,000.
Option 3 needs approval as a schema migration; it is not part of this
change.
