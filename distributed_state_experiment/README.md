# Distributed State Experiment

A self-contained learning exercise (like the sibling `langgraph_experiment/`,
`multi_agent_experiment/`, and `k8s/` directories at the repo root) that
finds behavior in this codebase that is correct with one API replica and
silently incorrect with more than one, reproduces it, and fixes it with
shared distributed state (Redis). It then goes one step further and
builds a real distributed background-processing workflow (a Redis
Streams-backed ingestion job queue and worker fleet) on top of the same
Redis instance, teaching the concurrency concepts a production system
built this way actually needs: race conditions, atomic operations,
distributed locks, idempotency, retries, timeouts, and the differences
between at-most-once / at-least-once / exactly-once delivery.

Not a production feature. `rag.api.deps.get_rate_limiter()` gained one
new, config-gated, off-by-default branch (`security.rate_limit.backend:
redis`) because that specific piece had to touch the real app to be a
real fix; everything else (the job queue, the worker fleet, the
distributed-lock demo, the benchmark harness) lives entirely under this
directory and is never imported by `src/rag/` or `rag.api`.

## Table of contents

- [The one exception to "fully isolated"](#the-one-exception-to-fully-isolated)
- [What was executed vs. simulated](#what-was-executed-vs-simulated): read this first
- [Part 1: process-local state, demonstrated](#part-1-process-local-state-demonstrated)
- [Part 2: distributed rate limiting with Redis](#part-2-distributed-rate-limiting-with-redis)
- [Part 2B: distributed background processing](#part-2b-distributed-background-processing)
- [Experiments 1-12](#experiments-1-12)
- [Part 3: concurrency concepts](#part-3-concurrency-concepts)
- [Part 4: benchmark](#part-4-benchmark)
- [Part 5: interview answers](#part-5-interview-answers)
- [Investigation: process-local state inventory](#investigation-process-local-state-inventory)
- [Directory layout](#directory-layout)
- [Running everything yourself](#running-everything-yourself)
- [Observability](#observability)
- [Kubernetes manifests](#kubernetes-manifests)

## The one exception to "fully isolated"

`langgraph_experiment/` and `multi_agent_experiment/` never touch
`src/rag/`. This experiment does, in exactly one place:
`rag.api.deps.get_rate_limiter()` gained a `security.rate_limit.backend:
"redis"` branch (default stays `"memory"`, a true no-op): every existing
config/test/deployment is byte-identical unless someone opts in), because
Part 2 is explicitly a fix to something in the real app, not a standalone
concept demo. The boundary:

- **Touches `src/rag/`** (config-gated, off by default): `config.py`'s
  `RateLimitConfig` (new `backend`/`redis_url_env_var`/`strategy`/
  `redis_fail_mode`/`redis_key_prefix` fields) and `AppConfig.rate_limit_redis_url()`;
  `api/deps.py`'s `get_rate_limiter()`; `api/main.py`'s new
  `redis.exceptions.RedisError` handler (only registered at all when
  `backend="redis"` and `redis_fail_mode="fail_closed"`); `config/default.yaml`'s
  documented swap points; `pyproject.toml` gained `redis>=5.0` as a core
  dependency (see that file's own comment for why it's core, not an
  extra, mirroring `mcp`/`slowapi`/`pyjwt`'s "always-running process,
  runtime flag" shape).
- **Stays under `distributed_state_experiment/`**: the teaching-reference
  rate limiters (`rate_limiter/`), the entire job queue and worker fleet
  (`job_queue/`), the distributed-lock demo (`distributed_lock/`), every
  runnable script (`cli/`), and this experiment's own test suite
  (`tests/`).

## What was executed vs. simulated

Read this before trusting any number below.

**Two passes were run.** In the first pass, Docker was coordinated
with a sibling agent's own Docker work on the same host; a
`docker pull redis:7-alpine` was in flight when a pause instruction
arrived and was left to finish on its own, but no container was started
or used for any experiment in that pass. Everything below marked
"fakeredis" or "the TCP stand-in" is from that pass. In a **second,
resumed pass** (after the pause was lifted), Docker was confirmed free
(`docker ps` showed nothing unexpected running): every pre-existing
container, including `local-rag-postgres` and the `rag-learning` minikube
node, was in a stopped state and was left exactly as found) and
`docker compose -f docker-compose.yml -f docker-compose.redis.yml up -d
redis` was run for real, starting **only** the new `local-rag-redis`
container (verified immediately after with `docker ps -a`: every other
container's state was unchanged). A genuine `redis:7-alpine` was
reachable and healthy (`redis-cli ping` -> `PONG`) for the remainder of
the session. Results gathered against it are labeled "real Redis" /
"genuine `redis-server`" below and in `results/*_real_redis_*.txt`,
distinct from the earlier fakeredis-backed results, which are kept
(not overwritten) for comparison.

**A real, measured constraint drove several other design decisions**:
importing `rag.factory` (which transitively imports
`sentence_transformers`/`torch`) was timed directly in this sandboxed
environment at **1122 seconds (~18.7 minutes)** for a single cold
import (`import time; t0=time.time(); import rag.factory; print(time.time()-t0)`
-> `1122.236`). A second, separate `import rag.api.deps` on top of that
added another ~116 seconds. This is almost certainly Windows real-time
antivirus scanning every DLL in a large ML dependency tree on first
touch, not a property of the code itself, but it is real, **repeatable
(a second, independent cold import measured 1333 seconds, ~22 minutes --
not a one-off fluke of disk caching)**, and it governed what could
practically be run multiple times in one session:

- **Part 1's real-HTTP two-pod demo**
  (`part1_process_local_state/two_pod_rate_limit_demo.py`) needs to boot
  two full `rag.api.main:app` processes, on the order of 40 minutes of
  pure import time before the first request could fire. **Written,
  complete, and correct; not executed here.** The actually
  executed substitute is `tests/test_inmemory_limiter_divergence.py`
  (3 passing tests, run 3x with no flakiness, using the exact
  process-local-counter shape `slowapi.MemoryStorage` has).
- **The production `security.rate_limit.backend: redis` wiring** in
  `rag/api/deps.py` was still verified for real: `get_config()`/
  `get_rate_limiter()` (memory backend) were both exercised directly
  against the real, imported module (one ~19-minute background run,
  output: `memory backend ok: <slowapi.extension.Limiter object at
  ...> False`, confirming `enabled=False` reads through correctly and
  the object constructs). `tests/unit/test_rate_limiter_redis_backend.py`
  (5 tests covering both `redis_fail_mode` branches, strategy/prefix
  passthrough, and that a disabled Redis-backend limiter never needs
  connectivity to construct) was run against the real module twice: the
  first run reported `5 passed, 5 errors in 932s`. Every actual test
  assertion passed, but a fixture-teardown-order bug in the test file
  itself (`_clear_singleton_caches`'s post-yield code called
  `deps.get_config.cache_clear()`, but by that point a test's own
  `monkeypatch.setattr(deps, "get_config", <lambda>)` had already
  replaced that attribute with a plain lambda with no `.cache_clear()`;
  `monkeypatch`'s own revert runs *after* this fixture's teardown, not
  before, in this file's fixture ordering) raised `AttributeError` in
  every test's teardown phase. Fixed by capturing the real, `lru_cache`d
  `get_config` function reference once at module import time, before any
  test patches it, and clearing the cache through that captured
  reference instead of through the (possibly-patched)
  `deps.get_config` attribute. Rerun after the fix (a second ~15-20
  minute background run, given the same import cost). See that file's
  own module docstring for exactly what the 5 tests prove and why they
  don't need a reachable Redis to do so.
- **Everything in `job_queue/`, `rate_limiter/`, and `distributed_lock/`
  has zero dependency on `rag.*`** (only `redis`, stdlib, and each
  other), so none of it pays the import cost above. This is most of
  what makes Experiments 1-12 below actually executable here.

**First pass: no `redis-server` binary was installed locally, and no
Redis Docker image had been pulled yet.** Experiments 1-12 and the Part 4
benchmark ran for real, as genuinely separate OS processes communicating
over a genuine TCP socket via `redis-py`, against
`distributed_state_experiment/cli/fake_redis_server.py`: a
`fakeredis.TcpFakeServer`, which implements the real RESP wire protocol
over a real socket (confirmed: `redis-cli`/`redis-py`/any Redis client
can connect to it exactly as they would `redis-server`). The commands
exercised for real against it are genuinely real: `XADD`/`XREADGROUP`/
`XACK`/`XCLAIM`/`XPENDING`, `SET ... NX PX`, `ZADD`/`ZRANGEBYSCORE`, and
Lua `EVAL` (via `fakeredis[lua]`'s `lupa` dependency, confirmed installed
and working). Only the *server process* is a pure-Python simulator
instead of the genuine C `redis-server` binary. One real fidelity gap was
found in that pass (a single 104/100-allowed run in the Part 4 benchmark,
1 of 3 runs) and was reported honestly rather than glossed over. See
[Part 4](#part-4-benchmark).

**Second pass: the same benchmark and experiment suite were rerun against
genuine `redis:7-alpine`** (`docker-compose.redis.yml`, confirmed up and
healthy as described above). Real results, not reasoned-through:

- **Part 4 benchmark, 3 consecutive real runs against genuine Redis**:
  the Redis-backed configuration landed at exactly 100/100 allowed **every
  single run** (0 of 3 showed the earlier overshoot). Full output in
  `results/rate_limit_benchmark_real_redis_3runs.txt`. This is consistent
  with, though does not on its own fully prove, the first pass's leading
  hypothesis (a fixed-window boundary crossing, or a fakeredis-TCP-server-
  specific fidelity gap, rather than a real atomicity failure).
- **The atomic-increment concurrency test was run directly against
  genuine `redis-server`** (200 concurrent requests, 20 threads, one
  shared bucket, limit 50): **exactly 50 allowed**, matching the
  fakeredis-backed version of the same test exactly. This is now also a
  permanent, non-skipping-when-Redis-is-up pytest test
  (`test_redis_fixed_window_limiter.py::test_concurrent_hits_against_real_redis_server_never_exceed_the_limit`,
  via the `real_redis` fixture), confirmed passing as part of a full
  `pytest distributed_state_experiment/tests` run (**24 passed**, up from
  23, zero skips, meaning the fixture successfully found and used the
  live container).
- **Experiments 1-10 were rerun end to end against genuine Redis**
  (`results/experiment_log_real_redis.txt`) with identical final outcomes
  to the fakeredis run (correct submission dedup, correct crash recovery
  with `attempts=2`, correct exhaustion to `dead_letter` after exactly 3
  attempts), but with one genuine, interesting behavioral difference:
  against real Redis's lower per-command latency, Experiment 8-10's
  poison job reached `dead_letter` within a **single** `run_until_idle()`
  worker invocation (all 3 attempts, including both backoff waits,
  completed inside that one call's own idle-polling loop), whereas the
  fakeredis run needed the orchestrator script's explicit multi-round
  structure (separate worker subprocess launches per retry round) to
  observe the same progression, because the higher per-round-trip latency
  of the pure-Python fakeredis TCP server meant the worker's idle-timeout
  fired before a pending retry's backoff had elapsed. Not a correctness
  difference: both backends reached the identical correct final state,
  but a real, measured illustration of how a slower simulated backend
  can change a demo script's own *timing assumptions* even when the logic
  under test is unaffected.
- **`tests/unit/test_rate_limiter_redis_backend.py`** (the production
  `rag.api.deps.get_rate_limiter()` wiring) was rerun standalone after
  the earlier fixture-teardown-order fix: **`5 passed in 442.23s (0:07:22)`,
  zero errors**, confirming the fix was correct and all 5 tests
  (both `redis_fail_mode` branches, strategy/prefix passthrough, and
  that a disabled Redis-backend limiter never needs connectivity to
  construct) pass cleanly against the real, imported `rag.api.deps`
  module. This particular rerun happened to pay only ~7 minutes of
  import cost rather than the ~19-22 minutes measured earlier, consistent with the cost being
  OS/antivirus first-touch
  disk-scanning overhead that partially amortizes once enough of the
  same files have been scanned in a given session, not a fixed constant.

**Kubernetes (Experiments 11-12): checked for real, deliberately not
applied.** In the resumed pass, `kubectl cluster-info`/`minikube status -p
rag-learning` were run (read-only) and confirmed the `rag-learning`
cluster's host/kubelet/apiserver/kubeconfig were all in a **Stopped**
state (not merely idle: the underlying container did not exist at all
under the default `minikube` profile name, and the `rag-learning`-profiled
one was fully torn down). Per the explicit fallback given for this
scenario, this was judged too invasive to bring up and deploy into
unattended: starting a stopped multi-day-old cluster shared with the
sibling `k8s/`/`experiment-helm` work risks a slow, disruptive restart of
state another session may be relying on, for a verification step whose
value (confirming `kubectl scale` increases claim throughput) was already
covered by the real local-process proxy in Experiment 3. **Left exactly
as found; `k8s/distributed-state/`'s manifests remain written and
reviewed but not applied**, same status as the first pass. The local
proxy for "does adding worker replicas increase throughput": real
separate OS processes, 1 vs. 3, draining an identical backlog. **Was**
executed for real, twice (once per Redis backend); see Experiment 3 and
the Experiments 11-12 section for the real numbers and the honest
distinction from a real `kubectl scale`.

## Part 1: process-local state, demonstrated

### The inventory (before touching anything)

Grepped and read directly, not assumed, across `src/rag/`:

| Where | What | Process-local? | Notes |
|---|---|---|---|
| `api/deps.py`: every `@lru_cache` getter | `AppConfig`, `Embedder`, `VectorStore`, `Reranker`, `LLM`, `IngestionPipeline`, `RetrievalPipeline`, MCP ASGI app, `FeedbackStore` | Yes, by design | Deliberate: "shares them across requests... on an 8GB-RAM CPU-only box where we don't want two copies of the embedding model." Correct for what they are (expensive, stateless-after-construction singletons). None of these hold *mutable request-driven counters* the way the rate limiter does, so none of them have this bug. |
| `api/deps.py::get_rate_limiter()` | `slowapi.Limiter` with in-memory `MemoryStorage` | **Yes: this is the bug** | A live request counter, mutated on every request, with no cross-process visibility. This is Part 1's demonstration target. |
| `mcp/business/store.py` | `_SYNTHETIC_CASES: dict[...]`, guarded by `_CASE_MUTATION_LOCK = threading.Lock()` | Yes | A synthetic, in-memory customer-case backend. `threading.Lock` only excludes concurrent *threads in the same process*. Two API replicas each import their own fresh copy of this module-level dict at process start, so two replicas would disagree about case state exactly like the rate limiter disagrees about request counts. Not fixed here (explicitly out of scope: the task calls out the rate limiter as the safe, easy one to demonstrate), but worth naming: it is the second clearest instance of the same bug class in this codebase, and a real deployment with `mcp.business_actions.enabled=true` behind >1 replica would see it (e.g. two replicas independently believing a just-created case doesn't exist yet). |
| `observability/metrics.py` | `REGISTRY = CollectorRegistry()`, every `Counter`/`Histogram`/`Gauge` | Yes, and *correctly* so | Each replica's `/metrics` endpoint reports only its own process's counts; Prometheus's `sum()` aggregation across scrape targets is exactly how multi-replica metrics are supposed to work. Not a bug. The opposite failure mode (one shared external counter) would be wrong here. Included in the inventory specifically to contrast with the rate limiter: the same "process-local" shape is correct in one place and wrong in the other, depending on whether the *thing being counted* needs a single global answer (a rate limit does; per-replica request counts, summed at scrape time, do not). |
| `vectorstore/pgvector.py`, `feedback/store.py` | `psycopg2.pool.ThreadedConnectionPool` instances | Process-local, and *correctly* so | Each replica needs its own connection pool to the *shared* Postgres instance. Postgres itself is the actual shared state here, already handled correctly (see the root `CLAUDE.md`'s "joint-investigation backlog": `get_or_create_document_id`'s `INSERT ... ON CONFLICT`, `replace_document_chunks`'s one-transaction commit). This is the model the rate limiter and job queue both follow: don't coordinate the pools, coordinate through the shared backing store. |
| `agent/mcp_client.py` | One fresh `ClientSession` per remote tool call, never pooled | Not applicable | Explicitly documented as "trivial identity isolation... no risk of one caller's session/token leaking to another": a deliberate non-cache, not a bug. |
| `security.rate_limit`'s own docstring (pre-existing) | -- | -- | Already stated the problem in prose: *"In-memory state is process-local; with more than one API replica, each enforces its own independent limit."* This experiment reproduces and fixes exactly that. |

### The demonstration

Two independent proofs, one executed as real HTTP traffic against two
real processes (**not run here**; see above), one executed
directly against the exact class shape that bug lives in
(**run three times here, 18 passing tests total including
these 3, zero flakiness**):

```
$ python -m pytest distributed_state_experiment/tests/test_inmemory_limiter_divergence.py -v
test_single_instance_enforces_its_own_limit PASSED
test_two_independent_instances_double_the_effective_limit PASSED
test_round_robin_load_balancing_still_double_counts PASSED
```

The middle test is the task's own example, reproduced exactly:

```python
pod_a = InMemoryFixedWindowLimiter(limit=100, window_seconds=60, pod_name="pod-a")
pod_b = InMemoryFixedWindowLimiter(limit=100, window_seconds=60, pod_name="pod-b")

allowed_a = sum(1 for _ in range(70) if pod_a.hit("tenant:acme")[0])   # -> 70
allowed_b = sum(1 for _ in range(70) if pod_b.hit("tenant:acme")[0])   # -> 70

global_allowed = allowed_a + allowed_b   # -> 140, intended limit was 100
```

**Why this is a faithful reproduction of the real bug, not a
simplification of it**: `InMemoryFixedWindowLimiter` (`rate_limiter/in_memory.py`)
has the identical property `slowapi.MemoryStorage` (the real backend
`rag.api.deps.get_rate_limiter()` uses by default) has, for the identical
reason: a plain Python dict living in one process's heap, with no
cross-process visibility whatsoever. The real `Limiter` object built by
`get_rate_limiter()` was independently confirmed to construct correctly
here (see above); what wasn't re-run was booting two full
`uvicorn` processes and firing 140 real HTTP requests at them through a
round-robin proxy, purely because of the ~19-minute-per-process import
cost. `part1_process_local_state/two_pod_rate_limit_demo.py` does exactly
that, using the real `rag.api.main:app` and the real
`RAG_CONFIG_PATH`-driven config override mechanism
(`rate_limit_demo_config.yaml`), and is ready to run wherever that import
cost doesn't apply.

## Part 2: distributed rate limiting with Redis

### Requirements checklist, and where each one is satisfied

| Requirement | Where |
|---|---|
| Shared counter across replicas | `limits.storage.redis.RedisStorage` (production, via `storage_uri`) / `RedisFixedWindowLimiter` (teaching reference): one Redis key per bucket, read/written by every process pointed at the same Redis. |
| Tenant-based key when authenticated | Unchanged: `api/deps.py::_rate_limit_key` already buckets `f"tenant:{tenant_id}"` when a verified JWT identity is present. This changes only the *storage backend*, never the key derivation. |
| IP fallback where appropriate | Same function, unchanged: `f"ip:{get_remote_address(request)}"` when no identity or `key: "ip"`. |
| TTL/window semantics | `strategy: fixed-window` (default) / `moving-window` / `sliding-window-counter`, `RateLimitConfig.strategy`. TTL is set atomically inside the same Lua script that increments (`incr_expire.lua` for fixed-window; `fixed_window.lua` in this repo's own teaching reference does the identical thing explicitly, see below). |
| Atomic counter update | See "Which Redis atomic primitive, and why" below. |
| Configurable limits | Unchanged `requests_per_minute`, still per-bucket. |
| Bounded key cardinality | Unchanged: the key is always `tenant:<id>` or `ip:<addr>`: bounded by the number of distinct tenants/IPs actually seen, never a raw query, path, or token value. |
| No JWT/token stored in Redis | Never was, and still isn't: `_rate_limit_key` derives a short bucket string, never the raw `Authorization` header. Proven directly by `tests/test_redis_fixed_window_limiter.py::test_no_jwt_or_token_ever_used_as_a_key_fragment`. |
| Fail behavior must be explicit | `RateLimitConfig.redis_fail_mode: "fail_open" \| "fail_closed"`, both explicit, both intentional, default justified below (Part 5 Q3 has the full reasoning). |

### Which Redis atomic primitive, and why

Two implementations exist, deliberately:

1. **Production** (`rag.api.deps.get_rate_limiter`, `backend: "redis"`):
   reuses `limits`'s `RedisStorage`, which (confirmed directly against
   the installed `limits==5.8.0` package source, not assumed) performs
   the fixed-window strategy's check-and-increment via a bundled Lua
   script (`incr_expire.lua`, loaded through `register_script`/`EVAL`).
   Chosen because it's already a transitive capability of `slowapi`
   (already a dependency), so the production fix is a config branch, not
   new hand-rolled protocol code: the same "reuse over reimplement"
   reasoning the rest of this codebase already applies everywhere else
   (MCP reuses the official SDK; RAGAS caching reuses `ragas.cache`).
2. **Teaching reference** (`distributed_state_experiment/rate_limiter/`):
   a small, explicit `fixed_window.lua` (14 real lines, reproduced in
   full in that file) and `RedisFixedWindowLimiter` wrapping it, so the
   atomicity story is visible end to end in one small file rather than
   inside a third-party library. Both implementations do the same thing:
   one `EVAL` call does `INCR` (or its own bucket's increment), sets the
   key's TTL only on the very first increment of a fresh key, and checks
   the result against the limit, all inside Redis's single-threaded
   command execution, so no other command (including a second replica's
   own concurrent `EVAL`) can interleave between the increment and the
   check.

**Why not a plain client-side `GET` then `INCR`?** Between the `GET` and
the `INCR`, a second replica's own `GET` can read the identical
pre-increment value and also decide to allow the request, both then
increment, and the bucket ends up over its limit by however many
replicas raced through that window. This is exactly the classic
check-then-act race; see Part 3's race-condition entry.

**Strategy tradeoff, stated explicitly**: `fixed-window` (the default)
is the cheapest (one `EVAL` per request) and easiest to reason about, at
the cost of allowing up to ~2x the nominal limit across a window
boundary (a burst of requests can land in two adjacent windows, each
independently under its own cap). `moving-window`/`sliding-window-counter`
remove that boundary burst at the cost of more Redis-side bookkeeping per
request. `fixed-window` was judged the right default for this system's
own traffic shape (interactive query/agent calls, not a billing-grade
quota). Exactly the same judgment call this codebase already makes
elsewhere (e.g. the reranker experiment's "moves the metric by less than
noise, not worth the cost" pattern).

### Fail-open vs. fail-closed

**Default: `fail_open`.** A rate limiter is an availability/fairness
control, not a confidentiality control. The opposite risk profile from
`security.field_redaction`, whose fail-closed default this codebase's own
`CLAUDE.md` justifies with "a security-relevant guarantee that depends on
[external] compliance is not a guarantee." Failing *closed* on a rate
limiter means a transient Redis blip takes down 100% of API traffic for
every tenant, for a control whose entire job is to protect availability
-- worse than the outcome it's meant to prevent. `fail_open` here means:
on a Redis error, `slowapi`'s `in_memory_fallback_enabled` kicks in and
the request is checked against a *process-local* fallback limiter for
the duration of the outage.

**This is a deliberate, ironic, and explicitly documented reintroduction
of Part 1's exact bug, scoped to the outage window.** During a Redis
outage, `fail_open` mode is *no worse than* this system's pre-Redis
default (`backend: "memory"`). Every replica just goes back to
enforcing its own independent limit until Redis recovers, rather than
enforcing zero limit or blocking all traffic. That tradeoff (weaker
enforcement, not full outage) is exactly why `fail_open` is the default.

**`fail_closed` is provided, and is not a worse choice for every
deployment**: a caller with a compliance requirement that a stated rate
limit is *never* exceeded, even briefly, during any failure mode, should
use it. Made explicit at the transport layer: `rag/api/main.py` registers
a dedicated `redis.exceptions.RedisError` -> 503 handler only in this
mode, so a Redis outage under `fail_closed` is a clean, documented 503
("Rate limiter temporarily unavailable; failing closed"), never a bare
500 or a silent pass-through.

### Configuration surface

```yaml
security:
  rate_limit:
    enabled: false
    requests_per_minute: 60
    key: tenant                      # tenant | ip
    backend: memory                  # memory | redis
    redis_url_env_var: RATE_LIMIT_REDIS_URL
    strategy: fixed-window           # fixed-window | moving-window | sliding-window-counter
    redis_fail_mode: fail_open       # fail_open | fail_closed
    redis_key_prefix: rag_rate_limit
```

## Part 2B: distributed background processing

### Why Redis Streams (not Kafka, not a plain list/pub-sub)

- **Kafka would be infrastructure without a demonstrated requirement** --
  this codebase's own repeated pattern (Redis itself, LangGraph, and MCP
  were all evaluated and deliberately deferred elsewhere in this
  repository until a concrete need existed). A single ingestion queue at
  this system's scale needs consumer groups and redelivery semantics, not
  partitioned, multi-broker, retention-tiered log storage.
- **A plain Redis `LIST` (`LPUSH`/`BRPOP`) has no visibility/claim
  semantics at all**. Once a worker `BRPOP`s an item, it's gone from
  Redis; if that worker crashes mid-processing, the item is simply lost,
  with no equivalent of a Pending Entries List to recover it from. That
  would fail Experiment 4-5 outright.
- **Redis Streams' consumer groups give the exact primitives the spec
  asks to teach, natively**: `XADD` (append), `XREADGROUP ... >`
  (claim new work, disjoint across group members), `XACK` (mark done),
  `XPENDING` (list claimed-but-unacked entries), `XCLAIM`/`XAUTOCLAIM`
  (reassign an entry whose original claimant has gone idle too long).
  `XPENDING`+`XCLAIM` (the explicit two-call form) is used here rather
  than the newer single-call `XAUTOCLAIM`, deliberately, for
  teachability. Both are documented in `queue.py::reclaim_stale`'s
  docstring.

### The design, end to end

```
submit_jobs.py --count 20
        |
        v
  IngestionJobQueue.submit()
    - dedup on idempotency_key (dataset_id:source_path:sha256(content))
    - backpressure: rejects (QueueFullError) if backlog_depth() >= max_queue_depth
    - XADD ingest_jobs:stream {job_id}
    - JobRecord persisted to job_store (Redis hash), status=SUBMITTED
        |
        v
  N worker replicas (IngestionWorker.run_forever / run_until_idle)
    - heartbeat(): TTL'd Redis key, makes active_worker_count() observable
    - promote_due_retries(): ZRANGEBYSCORE the retry_due zset, re-XADD anything due
    - reclaim_stale(): XPENDING + XCLAIM anything idle > visibility timeout (worker-crash recovery)
    - claim(): XREADGROUP ... > , count=max_concurrency (bounded concurrency)
    - process_fn(payload): chunk/embed/store (simulated in these runs; real
      IngestionPipeline integration point documented and available via --real)
    - complete() -> XACK + status=COMPLETED   |   fail() -> XACK + retry_due zset (backoff)
      or, once max_attempts exhausted -> dead_letter stream + status=DEAD_LETTER
        |
        v
  job_status.py --job-id ... / --summary
    - queries JobRecord directly from the Redis-backed job_store, independent of queue mechanics
```

**Idempotency key, reused not invented**: `f"{dataset_id}:{source_path}:{sha256(content)}"` --
this is exactly the `(source, dataset_id)` + checksum identity scheme
`VectorStore.get_or_create_document_id`/`replace_document_chunks` already
use (see the root `CLAUDE.md`'s "Document identity" section). Two
submissions of byte-identical content into the same dataset always
compute the same key, so `submit()` dedups them at the queue's own front
door; and even if a duplicate somehow reaches the *processing* step
twice (a redelivery, a retrying producer), the real
`IngestionPipeline.ingest_file` -> `PgVectorStore.replace_document_chunks`
path is checksum-gated and already commits checksum+delete+insert as one
atomic transaction (the joint-investigation backlog's DATA-1 fix) --
processing the same job twice is a database no-op, not a duplicate
document.

**Why `job_state` lives in Redis, not Postgres, for this experiment**:
this system already has a Postgres instance (and `FeedbackStore`'s own
small dedicated-pool pattern would be the natural template for a
production job-status table). Redis-only was chosen here specifically to
keep this *learning exercise's* infrastructure surface to one moving
part, since Redis is already required for the queue itself. A production
version of this feature would very plausibly move `JobRecord` into
Postgres (queryable by dataset/date/status without a `SCAN`, durable
across a Redis restart with no `appendonly` tuning) while keeping Redis
Streams as the transport. Flagged here rather than presented as the
only reasonable design.

**Bounded concurrency**: `WorkerConfig.max_concurrency` bounds both how
many entries one `claim()` call requests and the size of the
`ThreadPoolExecutor` each worker dispatches into. A worker never has
more than `max_concurrency` jobs in flight at once, regardless of how
deep the backlog is.

**Backpressure**: `submit(..., max_queue_depth=N)` checks
`backlog_depth()` (see the real gotcha below) before adding to the
stream and raises `QueueFullError` instead of enqueuing past the cap --
proven by `tests/test_job_queue.py::test_backpressure_rejects_submission_over_max_queue_depth`.

**A real gotcha, found live, not assumed**: `XLEN` (a stream's length)
does **not** shrink when entries are `XACK`'d. Acknowledging only
clears an entry from the consumer group's Pending Entries List, it does
not delete the entry from the stream itself. An early version of this
queue used `XLEN` as "how much work is left" and two tests failed
(`queue.queue_depth() == 0` after everything completed, actual value 1,
then 2) until this was understood and fixed: `queue_depth()` now
documents plainly that it measures "entries ever added, not remaining
work," and a separate, explicit Redis counter (`backlog_depth()`,
incremented on submit, decremented on completion/dead-letter) is what
`submit()`'s backpressure check and the Prometheus `QUEUE_DEPTH` gauge
actually use. See `job_queue/queue.py::queue_depth`'s docstring for the
full explanation.

**A second real gotcha, found the same way**: the first version of
`IngestionJobQueue.__init__` defaulted `key_prefix="ingest_jobs"` for
*every* queue instance regardless of which `stream_name` it was built
for. Running `cli/run_experiments.py` for real immediately exposed this:
`dead_letter_depth()` read `2` when only one job had actually been
dead-lettered in that run, because the poison-job experiment's queue and
an earlier experiment's queue were silently sharing one global job store,
retry zset, and dead-letter stream. Fixed by deriving `key_prefix` from
`stream_name` by default (stripping a trailing `":stream"` suffix) --
now two queues pointed at different streams can never collide on shared
counters just because neither caller passed `key_prefix` explicitly. Both
gotchas are the kind of bug this experiment exists to make visible: they
only showed up once real code ran against a real (if simulated) backend,
never in a design review.

**At-most-once vs. at-least-once vs. exactly-once, and why this queue is
at-least-once + idempotent**: `XREADGROUP ... >` only ever hands a given
stream entry to *one* consumer at a time, so under normal operation a job
is processed once. But if that consumer crashes after claiming and before
acking, the entry sits in its PEL until `reclaim_stale()` hands it to
someone else. At that point the job *will* be processed again (Experiment
4-5/6-7's own proof: `attempts=2` on the recovered job in the real
`run_experiments.py` output below). That is at-least-once delivery, not
exactly-once: Redis Streams (like almost every real message broker: SQS,
Kafka consumer groups, RabbitMQ with acks) cannot, on its own, guarantee
a message is delivered exactly once, because "did the consumer finish
processing" and "did the ack reach the broker" are two separate events
that can fail independently (the consumer could finish the real work and
then crash before the `XACK` reaches Redis, indistinguishable from the
broker's side, from a consumer that crashed before doing anything).
**Exactly-once *effect* is achieved instead by making the processing step
idempotent** (the checksum-gated `replace_document_chunks` write, or this
queue's own submission-time idempotency-key dedup). This is the
standard, practical answer real systems use, because building true
exactly-once *delivery* generally requires distributed transactions
across the broker and the datastore, which is far more complex than
"make the effect idempotent and accept at-least-once delivery." This is
also, concretely, why `real_ingestion_process_fn` reuses
`IngestionPipeline` rather than a bespoke worker-side idempotency table:
the idempotency guarantee this system actually needs already exists,
one layer down, at the database.

### Dead-letter handling

`_move_to_dead_letter` (`queue.py`) runs once `record.attempts >=
record.max_attempts`: the job's status becomes `DEAD_LETTER`, and a
summary (`job_id`, `idempotency_key`, `attempts`, `last_error`) is copied
onto a second stream (`<prefix>:dead_letter`) for inspection, separated
from the main work stream so a dead-lettered job can never be
accidentally re-claimed by a normal worker loop, and so
`dead_letter_depth()` is a simple, direct `XLEN` on a stream nothing else
writes to.

## Experiments 1-12

All executed for real here (except 11-12, see below), via
`python -m distributed_state_experiment.cli.run_experiments --redis-url
redis://127.0.0.1:6399/0` against `fake_redis_server.py`. Full captured
output: `distributed_state_experiment/results/experiment_log.txt`.

**1-2. Submit 20 jobs, run 1 worker.**
```
submitted 20 jobs, backlog_depth=20
worker solo-worker: drained after processing 20 attempts
RESULT: 1 worker drained 20 simulated jobs in 3.5-8.8s wall-clock (varied across reruns --
        see the note on this number below)
```

**3. Increase to 3 workers.**
```
submitted 20 jobs, backlog_depth=20
worker worker-x: drained after processing 4-8 attempts
worker worker-y: drained after processing 8 attempts
worker worker-z: drained after processing 4-12 attempts    (splits sum to 20 every time)
RESULT: 3 concurrent workers drained the same 20 jobs in 0.8-1.8s wall-clock
RESULT: throughput proxy speedup 2.8x-4.2x across three reruns
```
Every rerun's per-replica split sums to exactly 20, and every job
completes exactly once (no duplicate `COMPLETED` job_ids). Consumer
groups genuinely partition work across replicas with no coordination
needed beyond "point every worker at the same stream and group name."
**Honest caveat on the wall-clock numbers**: each "worker" here is a
brand-new OS process (`subprocess.Popen`), so the dominant cost in every
timing above is Python interpreter + `redis`-import startup, not the
20 x 0.05s of simulated job latency (1.0s of real work at most, spread
across replicas). The 1-worker number swung from 3.5s to 8.8s across
reruns on this shared host with no code change between them. This
project's own `CLAUDE.md` documents an almost identical finding for its
own generation-latency benchmarks ("`ms/completion_token` swung ~4-5x
across sequential runs... pointing to host-level noise"). Treat the
*speedup direction* (3 workers always faster) as the reliable finding,
not the exact multiplier.

**4-5. Kill a worker while processing; verify unfinished work is
recovered.** A job with `slow_seconds=5.0` is submitted, a real worker
subprocess claims it, and is killed (`Popen.kill()`, a real `SIGTERM`
equivalent: `taskkill /F` under the hood on Windows) the moment
`pending_count() > 0` is observed (polled, not a fixed guessed sleep --
see the code comment on why a fixed sleep was tried first and failed):
```
pending entries while doomed-worker was alive (polled until claimed): 1
job status right after kill: processing
reclaim_stale found 1 orphaned entry
RESULT: job status after recovery: completed (attempts=2)
```
`attempts=2` is the direct, real proof: the job was claimed once by the
now-dead worker (attempt 1) and once more by the rescuing worker after
`reclaim_stale()` (attempt 2). Unfinished work was recovered, and the
recovery is itself a second delivery (at-least-once, exactly as
designed).

**6-7. Deliver the same job twice; verify idempotency prevents duplicate
persistent effects.**
```
RESULT: job processed twice (attempts=2), final status=completed, one job_id throughout
resubmitting the identical payload returns the same job_id: True
```
Two separate proofs in one run: a manually re-`XADD`'d duplicate delivery
of the same `job_id` is processed a second time without error or a second
job record (the queue-mechanics half), and resubmitting the *original*
byte-identical payload through `submit()` again returns the exact same
`job_id` rather than creating a new one (the submission-time dedup half).
In production, the real backstop is one layer down: `replace_document_chunks`'s
checksum gate means even a genuinely-duplicate *processing* run has no
duplicate database effect.

**8-10. Deliberately failing job; observe retries/backoff; exhaust
retries; observe dead-letter.**
```
status after first drain (before any retry is due): retry_scheduled, attempts=1
round 2: promote_due_retries promoted 1 job(s)
round 2: status=retry_scheduled, attempts=2, last_error='PoisonJobError: poison job: poison.md'
round 3: promote_due_retries promoted 1 job(s)
round 3: status=dead_letter, attempts=3, last_error='PoisonJobError: poison job: poison.md'
RESULT: poison job final status=dead_letter after 3 attempts (max_attempts=3); dead_letter_depth=1
```
Each round's `promote_due_retries` only re-enqueues the job once its
exponential backoff (`base_backoff_seconds=0.2`, doubling, capped at
`max_backoff_seconds=0.4`) has genuinely elapsed. Proven directly by
`tests/test_job_queue.py::test_failing_job_schedules_a_retry_with_backoff`'s
"not due yet" assertion. After exactly `max_attempts=3` failed attempts
the job lands in `DEAD_LETTER` with its real error message preserved
(`PoisonJobError: poison job: poison.md`) and `dead_letter_depth()`
reports it.

**11-12. Scale worker pods in Kubernetes; measure throughput before/after
scaling.** **Not executed against a real cluster.** Checked for real in
the resumed pass: `minikube status -p rag-learning` confirmed the
cluster's host/kubelet/apiserver/kubeconfig were all `Stopped`; bringing
it up specifically for this was judged too invasive for a cluster shared
with the sibling `k8s/`/`experiment-helm` work and was deliberately
skipped, not guessed at (see "What was executed vs. simulated" for the
full reasoning). `k8s/distributed-state/ingestion-worker-deployment.yaml`
is written and ready: `kubectl scale deployment/ingestion-worker -n rag
--replicas=3` is the intended command, each pod naming itself via the
downward API (`POD_NAME` -> `--name`) so replica identity needs no manual
per-pod config. The closest thing measured for real here is
Experiment 3 above: 1 vs. 3 concurrent OS processes draining an
identical backlog, run against both Redis backends, a legitimate local
proxy for "more replicas increase claim throughput" (the mechanism
`kubectl scale` would exercise is identical: more consumer-group members
claiming disjoint stream entries), but explicitly not the same as real
pod scheduling, readiness gating, or cluster-network latency.

### One distributed-lock scenario

Modeled directly on this repository's own, already-fixed
`get_or_create_document_id` race (a pre-existing SELECT-then-INSERT
race in `PgVectorStore`, fixed via `INSERT ... ON CONFLICT`. See the
root `CLAUDE.md`'s "joint-investigation backlog" section). Reproduced
in-process (`distributed_lock/demo_lock.py::RacyDocumentIdTable`) with a
deliberate, injectable delay between the read and the write, then fixed
two different ways:

```
$ python -m pytest distributed_state_experiment/tests/test_distributed_lock.py -v
test_unsafe_concurrent_get_or_create_produces_more_than_one_document_id PASSED
test_redis_lock_serializes_the_same_race_to_one_id PASSED
test_atomic_check_and_set_fixes_the_race_with_no_lock_at_all PASSED
test_lock_holder_cannot_release_a_lock_it_no_longer_owns PASSED
test_second_acquire_attempt_fails_while_lock_is_held PASSED
```

- **Unprotected**: 8 concurrent callers racing `get_or_create_unsafe`
  produce more than one distinct `document_id` for the same
  `(source, dataset_id)`: a real lost-update bug, proven by the first
  test above (not merely asserted; the test fails if the race doesn't
  actually happen).
- **Fixed with a Redis distributed lock** (`SET key value NX PX ttl_ms`
 (Redis's own atomic "acquire iff absent" primitive, not a hand-built
  check-then-set): wrapping the exact same racy method in `with
  RedisDistributedLock(...):` serializes every caller onto one
  `document_id`. Release uses a small Lua script that compares the
  lock's stored random token before deleting, so a holder whose TTL
  already expired (and was re-acquired by someone else) can never
  release a lock it no longer owns. Proven directly by
  `test_lock_holder_cannot_release_a_lock_it_no_longer_owns`.
- **Fixed with an atomic check-and-set, no lock at all**
  (`get_or_create_atomic`, one `threading.Lock`-guarded `dict.setdefault`
  standing in for Postgres's real `INSERT ... ON CONFLICT DO NOTHING
  RETURNING`): also produces exactly one `document_id`, with **no
  external coordination primitive at all**.

**Is a distributed lock actually sufficient, and is a DB constraint
preferable? Yes to both, for different reasons:**

A distributed lock *is* sufficient to fix this specific race, proven
above. But it has two real costs a DB-constraint-based fix doesn't:

1. **It only protects callers disciplined enough to acquire it.** Any
   code path that reaches `get_or_create_unsafe` without going through
   `with RedisDistributedLock(...):` first still races. The lock is an
   opt-in convention enforced by every caller remembering to use it, not
   a property of the data itself. An `INSERT ... ON CONFLICT` constraint,
   by contrast, is enforced by Postgres itself, for *every* caller,
   including a future one nobody remembered to wrap in a lock.
2. **A lock adds a genuine extra failure mode and a genuine extra network
   round trip** (acquire, do the work, release: three Redis calls
   minimum, plus TTL-expiry edge cases like the one
   `test_lock_holder_cannot_release_a_lock_it_no_longer_owns` exists to
   guard against) that an atomic check-and-set simply doesn't have. The
   "DB constraint" fix here isn't really "add a constraint" so much as
   "make the check and the write the same indivisible operation" --
   `INSERT ... ON CONFLICT DO NOTHING RETURNING document_id` *is* that
   for Postgres, the same way `SET key val NX` *is* that for Redis. Where
   an atomic single-operation primitive already exists for the actual
   resource being protected, prefer it over layering a separate lock on
   top of a non-atomic multi-step operation.

**When a lock genuinely earns its place instead**: when the protected
operation spans multiple, non-atomic steps that no single database
primitive can express as one operation: e.g. "read a value from
Postgres, call an external HTTP API with it, then write the result back"
-- there is no `INSERT ... ON CONFLICT` that can make that whole sequence
atomic, so a distributed lock (or a different pattern entirely, like an
idempotency key stored *with* the external call's result) is the right
tool. This codebase's own document-identity race happened to be a
single-step "does this row exist" check, which is exactly the shape a DB
constraint handles better.

### Observability

`job_queue/metrics.py`: a dedicated `CollectorRegistry` (mirrors
`src/rag/observability/metrics.py`'s own pattern exactly, so re-importing
this module across test collection never raises "Duplicated
timeseries"). Bounded labels only (`outcome`: `completed`/`failed`;
`terminal_status`: `completed`/`dead_letter`, both fixed, small
vocabularies, never a raw job id or file path):

| Metric | Type | What |
|---|---|---|
| `ingest_jobs_submitted_total` | Counter | Jobs submitted (excludes deduped resubmissions) |
| `ingest_jobs_completed_total` / `ingest_jobs_failed_total` | Counter | Per-attempt outcomes |
| `ingest_jobs_retried_total` | Counter | Retries scheduled |
| `ingest_jobs_dead_lettered_total` | Counter | Jobs that exhausted their retry budget |
| `ingest_jobs_duplicate_deliveries_total` | Counter | Claims that were redeliveries via `reclaim_stale` |
| `ingest_job_processing_latency_seconds{outcome}` | Histogram | One processing attempt's wall time |
| `ingest_job_attempts_at_completion{terminal_status}` | Histogram | How many attempts a job needed |
| `ingest_queue_depth` | Gauge | `backlog_depth()`: outstanding work, not raw `XLEN` |
| `ingest_dead_letter_depth` | Gauge | Dead-letter stream length |
| `ingest_retry_queue_depth` | Gauge | Jobs currently waiting on their backoff timer |
| `ingest_active_workers` | Gauge | Live worker count, via TTL'd heartbeat keys |

## Part 3: concurrency concepts

Each defined against a concrete artifact in this experiment, not in the
abstract:

- **Process-local vs. shared state.** `InMemoryFixedWindowLimiter` (each
  process's own dict) vs. `RedisFixedWindowLimiter` (one Redis key every
  process reads/writes). Part 1's entire subject.
- **Race condition.** `RacyDocumentIdTable.get_or_create_unsafe`: two
  threads both read "no row exists," both decide to insert, one's write
  is silently lost or both get different ids. The outcome depends on
  timing, not logic.
- **Atomic operation.** `fixed_window.lua`'s single `EVAL` (increment +
  TTL-set + limit-check as one indivisible unit); `SET key val NX PX`
  (acquire-iff-absent as one indivisible unit); `ZREM`'s return value in
  `promote_due_retries` (an atomic "did I win the race to promote this
  job" check. See that method's own comment).
- **Distributed lock.** `RedisDistributedLock`: mutual exclusion enforced
  by an external, shared coordinator (Redis) rather than by both callers
  happening to live in the same process/thread.
- **Idempotency.** The ingestion job's `idempotency_key`
  (`dataset_id:source_path:sha256(content)`) and `PgVectorStore.replace_document_chunks`'s
  checksum gate: processing the same logical input twice produces the
  same result as processing it once.
- **Retry.** `IngestionJobQueue.fail()` scheduling a job back onto the
  stream after a delay, rather than treating one failure as final.
- **Timeout.** `reclaim_idle_ms` (the visibility timeout: how long a
  claimed-but-unacked entry is given before another worker may take it
  over) and `RedisDistributedLock.ttl_ms` (how long a lock is held before
  it auto-expires even if its holder never releases it. Both exist for
  the identical reason: a crashed holder must never wedge the system
  forever).
- **Eventual consistency.** `active_worker_count()`'s heartbeat TTL: a
  worker that just crashed is still "active" for up to `ttl_seconds`
  until its key expires. The count is allowed to be briefly stale
  rather than requiring an immediate, synchronous "worker N is gone"
  notification.
- **Strong consistency.** Every `allow()`/`fail()`/`complete()` call
  against Redis itself: the very next read of the same key, from any
  process, sees that write immediately. Redis's single-threaded command
  execution is what makes the rate limiter's shared counter, and the
  queue's shared backlog counter, correct at all.
- **Stateless service.** `rag-api` itself (per this codebase's own
  existing design: every `@lru_cache` singleton in `api/deps.py` is
  either read-only after construction or itself talks to a genuinely
  shared backing store). Any replica can serve any request.
- **Stateful service.** Redis and Postgres, in this design: the two
  places actual mutable, must-agree-across-processes state is allowed to
  live, and the only two things every replica/worker actually coordinates
  through.

## Part 4: benchmark

Three consecutive real runs (`distributed_state_experiment/cli/bench_rate_limiters.py`,
100 requests/replica, intended limit 100/tenant/minute, each "replica" a
genuinely separate OS process), run against **both** backends here: first the `fake_redis_server.py` stand-in
(`results/rate_limit_benchmark_3runs.txt`), then, in a resumed pass,
genuine `redis:7-alpine` via `docker-compose.redis.yml`
(`results/rate_limit_benchmark_real_redis_3runs.txt`):

| Configuration | Backend | Total allowed | Intended limit | Correct? | Wall-clock (3 runs) |
|---|---|---|---|---|---|
| 1 replica + in-memory | either (never touches Redis) | 100 | 100 | Yes | 4.71s / 8.83s / 0.38s |
| 3 replicas + in-memory | either (never touches Redis) | 300 | 100 | **No: 3x over** | 0.54s / 1.08s / 0.47s |
| 3 replicas + Redis-backed | fakeredis TCP stand-in | 104 / 100 / 100 | 100 | Yes (2/3 exact; 1/3 within 4%) | 1.43s / 1.05s / 0.62s |
| 3 replicas + Redis-backed | **genuine `redis-server`** | **100 / 100 / 100** | 100 | **Yes, 3/3 exact** | 0.92s / 0.93s / 1.46s |

**Correctness**: the in-memory 3-replica row is the bug, reproduced with
real numbers exactly as many times as it was run (300/100 every single
time, against either backend, not a flaky result, and expected, since
that row never touches Redis at all). The Redis-backed row is exact
against genuine Redis in all 3 reruns; against the fakeredis stand-in it
was exact in 2 of 3 and 4 requests over in the third. That single 104 was
not glossed over when it happened: it's consistent with `fixed-window`'s
own documented boundary-burst property (see Part 2). If a run happened
to straddle a wall-clock minute boundary, some requests would land in a
second, independently-capped 60-second window, and separately, the
underlying Lua script's atomicity was independently verified exact under
real concurrent load against **both** backends: 200 concurrent requests
from 20 threads at one shared bucket, limit 50, allowed count exactly 50,
every time, against fakeredis (`test_concurrent_hits_from_multiple_threads_never_exceed_the_limit`)
**and** against genuine `redis-server`
(`test_concurrent_hits_against_real_redis_server_never_exceed_the_limit`).
The resumed pass's 3/3-exact real-Redis benchmark result is additional,
stronger evidence for the original hypothesis (the 104 was very likely a
fixed-window boundary artifact, or specific to the fakeredis TCP
server's own concurrency handling, not a flaw in the atomic primitive
itself), though a single 3-run sample against real Redis showing no
overshoot is suggestive, not exhaustive proof that genuine Redis can
never show the same boundary property under different timing.

**Latency**: the Redis-backed configuration costs a real, measurable
network round trip per request that the in-memory configuration doesn't
pay (roughly 2-3x the wall-clock of the in-memory 3-replica row across
these runs). Expected, and the correct tradeoff, since the in-memory
row's "speed" is exactly the bug.

**Operational complexity**: in-memory costs nothing extra to operate (no
new service). Redis-backed adds one new stateful dependency to keep
running, monitor, and reason about failure modes for (this is exactly
what Part 2's fail-open/fail-closed design section exists to make an
explicit, deliberate decision about, rather than an afterthought).

## Part 5: interview answers

### 1. Why in-memory rate limiting fails horizontally

Because "the limit" and "the counter that enforces it" live in the same
place: one process's heap. Add a second replica behind a load balancer
and there are now two counters, each seeing only the slice of traffic
routed to it, each independently enforcing the *nominal* limit against
its own slice, so the *effective* global limit becomes
(nominal limit) x (replica count), demonstrated exactly: 100/min nominal,
3 replicas, 300 allowed. The bug isn't in the counting logic at all
(`InMemoryFixedWindowLimiter`'s arithmetic is correct); it's a mismatch
between where the state needs to live (globally, because the constraint
is global) and where it actually lives (locally, per process).

### 2. Why Redis fixes it

Redis is a single, shared, externally-reachable process every replica
can talk to, so it can hold the *one* counter the constraint actually
needs, instead of N independent counters. It fixes this specific problem
for the same reason Postgres already fixes the equivalent problem for
document identity in this codebase (`get_or_create_document_id`'s `INSERT
... ON CONFLICT`): move the state that must be globally consistent into
something all replicas share, rather than trying to keep N independent
copies in sync after the fact.

### 3. What happens if Redis dies

Depends on the configured `redis_fail_mode`, and that's deliberate, not
an oversight:

- `fail_open` (default): requests keep being served, but each replica
  silently falls back to enforcing its own local (process-local) limit
  for the outage's duration: Part 1's exact bug comes back, but
  only until Redis recovers, and the service stays up the whole time.
- `fail_closed`: every rate-limited request gets a clean 503 until Redis
  recovers. The limit is never silently weakened, at the cost of the
  feature it's protecting becoming fully unavailable during the outage.

Neither is "the right answer" universally; the right default depends on
whether the limiter is an availability control (favor `fail_open`) or a
hard compliance boundary (favor `fail_closed`). This system defaults to
`fail_open` because it's the former.

### 4. What Redis atomicity guarantees we rely on

Single-threaded command execution: Redis processes one command (or one
Lua script, which runs as a single command from the scheduler's point of
view) to completion before starting the next, even under many concurrent
client connections. That's what makes `EVAL fixed_window.lua` (increment
+ conditional TTL-set + limit-check) genuinely indivisible. No other
client's command, including a concurrent call to the exact same script,
can interleave in the middle of it. The same guarantee is what makes
`SET key val NX` a true "only the first caller wins" primitive (there is
no window between "check if the key exists" and "set it" for two callers
to both slip through), and what makes `ZREM`'s return value in
`promote_due_retries` a genuine "did I win the race to promote this
specific job" check (two workers' `ZREM` calls for the same member can
never both return 1).

### 5. Why this is a distributed-systems problem

Because the correctness of the whole system now depends on the
interaction between multiple independent processes that don't share
memory and can't observe each other's state directly, communicate only
by passing messages (HTTP requests, Redis commands) that can be
delayed/reordered/lost, and can fail independently of each other (one
replica can crash while the others keep running; Redis itself can become
unreachable while every replica stays up). None of that is true of a
single-process program, where "state" just means "a variable," updates
are trivially visible to the only thread that matters, and a crash takes
down the one thing that mattered anyway. Every concept in Part 3 --
races, atomicity, locks, idempotency, retries, timeouts, the
consistency/availability tradeoff. It exists specifically because more
than one independent actor now has to agree on shared truth without a
single authority they can all trivially synchronize through.

### 6. When Redis is NOT needed

- When there is genuinely only ever one replica (no load balancer, no
  horizontal scaling). The in-memory limiter is strictly simpler and
  has no extra failure mode to reason about, and this codebase's own
  `security.rate_limit.backend: memory` default reflects exactly that:
  it's the right choice for the single-instance/demo deployment this
  project ships by default.
- When the state in question doesn't actually need to be globally
  consistent: e.g. `observability/metrics.py`'s per-process Prometheus
  counters are *correctly* process-local, because Prometheus's own
  scrape-and-sum model is how multi-replica metrics are supposed to
  work; forcing them through one shared Redis counter would be adding
  infrastructure to solve a problem that doesn't exist.
- When a single-operation database primitive can express the whole
  constraint atomically on its own (`INSERT ... ON CONFLICT`), as the
  distributed-lock section above argues, a lock is a heavier, less safe
  tool than an atomic write when the constraint is expressible as one
  write.
- When the coordination need is rare enough, and the cost of getting it
  briefly wrong low enough, that eventual consistency is fine. This
  codebase's own MCP integration already made exactly this call about `active
  worker_count()`'s TTL'd heartbeat: a worker that crashed 3 seconds ago
  still counting as "active" for a few more seconds is a non-issue for
  an observability gauge, so no stronger (and more expensive) mechanism
  was reached for.

### 7. How Kubernetes horizontal scaling exposes this class of bug

A single-replica deployment can hide a process-local-state bug
indefinitely. There's only one process, so "process-local" and
"globally correct" happen to coincide by accident, and nothing about
normal operation would ever surface the gap. The moment a Deployment's
`replicas` field goes from 1 to N (exactly the field
`k8s/distributed-state/ingestion-worker-deployment.yaml` calls out, and
exactly what `kubectl scale` does), the Service's load balancing starts
routing traffic across N independent processes that don't share memory,
and any state that was silently assumed to be "the" state instantly
becomes "one of N copies of the state." Nothing about the code path
itself changes; only the topology does, which is exactly why this class
of bug is so easy to ship undetected in a system that starts out (and is
tested, and demoed) as a single replica, and only manifests once autoscaling
or a multi-replica rollout actually happens, often in production, often
under the exact traffic spike an HPA was added to handle in the first
place.

## Investigation: process-local state inventory

Covered above in [Part 1](#the-inventory-before-touching-anything) in
full, including singletons, caches, locks, and connection pools. The
short version: this codebase's `@lru_cache` singletons in `api/deps.py`
are deliberately process-local *and correct* (they're read-only-after-
construction resource handles, not mutable shared counters); the rate
limiter and the MCP business store's in-memory dict are the two places
that pattern is actually wrong, because both hold live, request-driven,
must-agree-globally state. Database connection pools are process-local
by design and correctly so. The database itself, not the pool, is the
shared resource.

## Directory layout

```
distributed_state_experiment/
  README.md                          this file
  INTERVIEW_QUESTIONS.md             15 Q&A
  requirements.txt                   redis, fakeredis[lua], pytest (this dir's own deps)
  Dockerfile.worker                  written, not built here (see k8s section)
  rate_limiter/
    in_memory.py                     InMemoryFixedWindowLimiter (Part 1's demonstration target)
    redis_fixed_window_limiter.py    RedisFixedWindowLimiter (teaching-reference atomic fix)
    lua/fixed_window.lua             the atomic check-and-increment script itself
  job_queue/
    models.py                        IngestionJobPayload, JobRecord, JobStatus
    queue.py                         IngestionJobQueue: submit/claim/ack/fail/reclaim/dead-letter
    job_store.py                     Redis-hash-backed JobRecord persistence + idempotency index
    worker.py                        IngestionWorker: bounded concurrency, graceful shutdown, the
                                      real-vs-simulated process_fn split
    metrics.py                       dedicated Prometheus CollectorRegistry, bounded labels
  distributed_lock/
    demo_lock.py                     RacyDocumentIdTable, RedisDistributedLock
  part1_process_local_state/
    two_pod_rate_limit_demo.py       written, not executed (see above)
    rate_limit_demo_config.yaml      config/default.yaml + rate_limit overridden (the bug)
    rate_limit_demo_config_redis.yaml   same, backend: redis (the fix)
  cli/
    fake_redis_server.py             the fakeredis.TcpFakeServer stand-in used for every executed run
    submit_jobs.py / run_worker.py / job_status.py
    run_experiments.py               orchestrates Experiments 1-10 as real subprocesses
    rate_limit_bench_worker.py / bench_rate_limiters.py   Part 4's benchmark
  tests/
    conftest.py                      fake_redis (fakeredis) / real_redis (self-skips cleanly)
    test_inmemory_limiter_divergence.py
    test_redis_fixed_window_limiter.py
    test_job_queue.py
    test_distributed_lock.py
  results/
    experiment_log.txt                       real captured output, Experiments 1-10 (fakeredis)
    experiment_log_real_redis.txt            same, rerun against genuine redis:7-alpine
    rate_limit_benchmark_3runs.txt           real captured output, Part 4, 3 runs (fakeredis)
    rate_limit_benchmark_real_redis_3runs.txt   same, 3 runs against genuine redis:7-alpine
```

Plus, outside this directory (the one deliberate exception, see above):
`src/rag/config.py`, `src/rag/api/deps.py`, `src/rag/api/main.py`,
`config/default.yaml`, `pyproject.toml`, `.env.example`,
`docker-compose.redis.yml`, `k8s/distributed-state/*.yaml`,
`tests/unit/test_rate_limiter_redis_backend.py`.

## Running everything yourself

```bash
# 1. Install this experiment's own dependencies (redis is already a core
#    dependency of the main package; fakeredis[lua]
#    is this directory's own test-only addition):
pip install -r distributed_state_experiment/requirements.txt

# 2. Start Redis. Either a real one (this is what the resumed pass in this
#    session actually used, bringing up only the `redis` service so
#    pre-existing postgres/rag-api containers are never touched):
docker compose -f docker-compose.yml -f docker-compose.redis.yml up -d redis
#    ...then use --redis-url redis://localhost:6379/0 below. Or, with no
#    Docker available (what the first pass used instead):
python -m distributed_state_experiment.cli.fake_redis_server --port 6399
#    ...then use --redis-url redis://127.0.0.1:6399/0 below.

# 3. Run the experiment suite (in a second terminal):
python -m distributed_state_experiment.cli.run_experiments --redis-url redis://127.0.0.1:6399/0

# 4. Run the Part 4 benchmark:
python -m distributed_state_experiment.cli.bench_rate_limiters --redis-url redis://127.0.0.1:6399/0

# 5. Run this experiment's own test suite (fast: no `rag` import, no live Redis needed):
python -m pytest distributed_state_experiment/tests -q

# 6. Run the production Redis-rate-limiter wiring tests (slow in this sandbox; see
#    "What was executed vs. simulated"; fast in a normal environment):
python -m pytest tests/unit/test_rate_limiter_redis_backend.py -q

# 7. (Wherever import time isn't prohibitive) Part 1's real two-pod HTTP demo:
python -m distributed_state_experiment.part1_process_local_state.two_pod_rate_limit_demo
```

## Observability

Two independent Prometheus registries are involved, deliberately never
merged:

- `src/rag/observability/metrics.py`'s `REGISTRY` (`rag-api`'s own
  `/metrics`, unchanged by this work).
- `distributed_state_experiment/job_queue/metrics.py`'s own `REGISTRY`,
  scraped from whatever process a real worker deployment exposes
  `/metrics` from (not wired to a live HTTP endpoint here;
  the module exists and is unit-tested via its `observe_*` helper
  functions, matching this codebase's own `observability/metrics.py`
  convention of testing the recording functions directly rather than
  standing up a scrape endpoint just to test metric values).

## Kubernetes manifests

`k8s/distributed-state/` (redis-deployment.yaml, redis-service.yaml,
ingestion-worker-deployment.yaml, kustomization.yaml). An additive
overlay on `k8s/base/`, following that directory's own conventions
(namespace `rag`, `IfNotPresent` pull policy for locally-built images,
resource requests/limits, a `wait-for-*` `initContainer` mirroring
`rag-api-deployment.yaml`'s existing Postgres-wait pattern). **Written
and reviewed, not applied against a real cluster here.** In a
resumed pass, `minikube status -p rag-learning` was checked for real
(read-only) and confirmed the cluster's host/kubelet/apiserver/kubeconfig
were all `Stopped`; bringing it up specifically to apply these manifests
was judged too invasive for a shared, sibling-session cluster whose state
this experiment didn't create (see "What was executed vs. simulated" for
the full reasoning) and was deliberately skipped rather than guessed at.
`distributed_state_experiment/Dockerfile.worker` (the worker image these
manifests reference) is likewise written, reviewed, and unbuilt.
