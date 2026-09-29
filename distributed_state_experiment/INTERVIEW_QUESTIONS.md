# Interview Questions: Distributed State Experiment

15 questions grounded in what was actually built and measured in
`distributed_state_experiment/` (see `README.md` for the full design and
real captured numbers this file's answers reference). Follows the same
Q&A study-guide style as `k8s/LEARNING.md`.

---

**1. Why did the in-memory rate limiter work fine in testing and demos, but become wrong the moment the service ran behind a load balancer with more than one replica?**

Because "correct" for a rate limiter means "the *global* count across
every caller stays under the limit," and an in-memory counter can only
ever see the requests that happened to land on its own process. With one
replica, "the requests that landed on this process" and "all the
requests" are the same set, so the bug is invisible by coincidence, not
because the logic is right. The measured proof: 1 replica + in-memory
allowed exactly 100 of 100 intended; 3 replicas + in-memory allowed 300
(100 per replica, each correctly enforcing its own local view of a limit
that was never actually shared).

---

**2. Walk through exactly why a plain `GET` then `INCR` against Redis is not safe for a shared rate limiter, even though each individual command is atomic.**

Each command *is* atomic on its own, but the race lives in the gap
*between* the two commands, which is not atomic. Replica A's `GET`
returns 99 (under the limit of 100); before A's `INCR` runs, replica B's
own `GET` also reads 99 (A hasn't incremented yet) and also decides to
allow the request. Both then `INCR`, and the bucket ends up at 101 with
two requests both believing they were the 100th allowed one. The fix used
here is folding the read, the conditional TTL-set, and the write into one
Lua script (`fixed_window.lua`), which Redis executes as a single,
uninterruptible unit -- no other command, including a second replica's
own concurrent call to the identical script, can run in the middle of it.

---

**3. What's the difference between the production fix in `rag/api/deps.py` and the `RedisFixedWindowLimiter` class this experiment also wrote? Why do both exist?**

The production fix reuses `limits`'s `RedisStorage` (already a
capability of `slowapi`, a pre-existing dependency), which performs the
identical fixed-window atomic increment via its own bundled Lua script --
confirmed by reading the installed package's source, not assumed. That's
the "reuse over reimplement" choice this whole codebase already follows
everywhere else. `RedisFixedWindowLimiter` (and its 14-line
`fixed_window.lua`) is a small, standalone, from-scratch implementation
of the same idea, written specifically so the atomicity story is visible
and inspectable in one file for this experiment's own demos, benchmark,
and tests, independent of a third-party library's internals. Neither is
"the real one" and the other "the toy one" -- they solve the same
problem for two different audiences.

---

**4. You chose `fail_open` as the default when Redis is unreachable. Defend that choice, and describe a system where you'd choose `fail_closed` instead.**

A rate limiter protects availability and fairness, not confidentiality --
the harm from a brief over-limit burst during an outage is bounded and
recoverable; the harm from taking the entire API down because its
rate-limiting sidecar had a network blip is not proportionate to what
the control is actually for. `fail_open` here specifically means
degrading to a *process-local* fallback limiter for the outage's
duration (via `slowapi`'s `in_memory_fallback_enabled`), not "let
everything through unbounded" -- so the worst case during an outage is
exactly this system's own pre-Redis default behavior, not zero
enforcement. I'd choose `fail_closed` for a system where exceeding a
rate is a hard compliance or billing boundary (e.g. a paid API tier where
overage has real financial consequences, or a security-sensitive
operation-per-minute cap) -- there, briefly refusing all traffic during
an outage is a better outcome than silently letting a limit slip, and
that's exactly why `fail_closed` is implemented and configurable here
too, not just theorized about.

---

**5. Your Part 4 benchmark showed 104 allowed instead of 100 in one of three runs against the Redis-backed limiter. Is that a bug? How did you investigate it, and what did you actually conclude?**

I didn't wave it away, and I didn't claim more certainty than I had
either. Two things are true simultaneously: the fixed-window strategy has
a *documented*, expected property that a request stream straddling a
wall-clock window boundary can land up to (limit x 2) requests across two
adjacent, independently-capped windows -- so a 4% overshoot is consistent
with "the run happened to cross a minute boundary." Separately, I
*directly* verified the underlying atomic primitive itself never loses a
race: 200 concurrent requests from 20 real threads at one shared bucket
with limit 50, via the in-process `fakeredis` test, landed at exactly 50
allowed, every time. What I did *not* do is capture the exact timestamps
of that specific 104-run to confirm it actually crossed a window
boundary -- so my honest conclusion is "the most likely explanation is
the window-boundary property, the atomicity itself is separately proven
sound, and the specific cause of this one run wasn't run to ground,"
not "definitely explained" and not "definitely a bug." I later got a
chance to re-run the identical benchmark against a genuine `redis:7-alpine`
container instead of the fakeredis stand-in: 3 consecutive runs, all
3 landed at exactly 100/100, and the same atomic-increment concurrency
test rerun directly against real `redis-server` also came back exactly
50/50. That's stronger evidence for the original hypothesis (not proof
that real Redis can never show the same fixed-window boundary property
under different timing, but a real data point in that direction), and I
still don't overstate it: a 3-run sample with no overshoot doesn't rule
out the possibility of hitting it again on a different run.

---

**6. What's the difference between at-most-once, at-least-once, and exactly-once delivery, and which one does your job queue actually provide?**

At-most-once: a message might be delivered zero or one times, never more
-- simple, but work can silently be lost (e.g. a plain `BRPOP` off a
list, where a worker crash after popping loses the item permanently).
At-least-once: a message is delivered one or more times -- no work is
silently lost, but a consumer might see (and must handle) the same
message twice. Exactly-once: delivered precisely once, guaranteed --
extremely hard to achieve in the general case, because "the consumer
finished the work" and "the broker recorded that fact" are two separate
events across a network that can fail independently. This queue provides
at-least-once: Redis Streams' consumer groups guarantee a given entry is
never dropped (it survives in a Pending Entries List until explicitly
acked or reclaimed), but if a worker crashes after finishing the real
work and before its `XACK` reaches Redis, that job will be redelivered
and processed again -- proven directly in this session (a job recovered
via `reclaim_stale()` after a real process kill came back with
`attempts=2`).

---

**7. If your system only provides at-least-once delivery, how do you make sure a job that gets processed twice doesn't do something wrong twice?**

By making the processing step idempotent, so "processed twice" and
"processed once" produce the identical end state. Two layers here,
deliberately: the queue's own submission-time dedup (a
`dataset_id:source_path:sha256(content)`-keyed lookup means resubmitting
byte-identical content returns the same `job_id` instead of creating a
second job), and, more fundamentally, the real database write this queue
was modeled on -- `PgVectorStore.replace_document_chunks` already commits
a document's checksum-gated replace as one atomic transaction, so running
the exact same ingestion twice is a database no-op the second time. This
matters because true exactly-once *delivery* would require the broker and
the datastore to participate in a distributed transaction, which is far
more complex than accepting at-least-once delivery and making the effect
of processing idempotent -- the practical answer real systems (SQS,
Kafka consumer groups, this queue) converge on.

---

**8. Explain the difference between `XACK` and deleting a message from a Redis Stream. Why does this matter for a metric like "queue depth"?**

`XACK` only removes an entry from the consumer group's Pending Entries
List -- it marks "this consumer is done with this entry," but the entry
itself stays physically present in the stream (visible to `XRANGE`,
counted by `XLEN`) until it's explicitly trimmed (`XTRIM`/`MAXLEN`) or
deleted (`XDEL`). I hit this directly: an early version of this queue
used `XLEN` as "how much work remains," and two tests failed after jobs
that had already completed successfully, because `XLEN` had never
shrunk. The fix was a separate, explicit Redis counter
(`backlog_depth()`, incremented on submit, decremented on completion or
dead-lettering) for "outstanding work," keeping `queue_depth()` around
but re-documented as "total entries ever added," which is a genuinely
different, also-useful number (e.g. for noticing an unbounded stream
that's never being trimmed) but not the same thing.

---

**9. You found a bug where different job-queue instances were sharing state that should have been isolated. What was it, and how did you find it?**

`IngestionJobQueue.__init__` originally defaulted `key_prefix` to one
fixed literal, `"ingest_jobs"`, for every instance regardless of which
`stream_name` it was constructed with. That meant two logically separate
queues -- say, one experiment's throughput-test queue and a completely
different experiment's poison-job queue -- silently shared one job store,
one retry sorted set, and one dead-letter stream, just because neither
caller happened to pass `key_prefix` explicitly. I found it by actually
running the multi-section experiment script end to end against a real
(if simulated) Redis backend rather than only unit-testing each piece in
isolation: `dead_letter_depth()` read `2` in a run where only one job had
actually been dead-lettered in that section, which was the tell. The fix
was making the default `key_prefix` derive from `stream_name` instead of
being one global constant, so two queues pointed at different streams can
never collide just because a caller left an optional parameter unset.
This is a concrete example of why running real integration flows against
a real backend (even a fake one implementing the real protocol) surfaces
bugs that unit tests, each scoped to their own fresh mock, structurally
cannot.

---

**10. What is a "visibility timeout" in the context of a message queue, and where does it show up in your design?**

It's how long a queue waits after a consumer claims a message before
assuming that consumer might have died and making the message available
to someone else again. In this design it's `reclaim_idle_ms`
(`WorkerConfig`): an entry a consumer group member claimed via
`XREADGROUP` but hasn't yet `XACK`'d becomes eligible for `reclaim_stale()`
(implemented as the explicit `XPENDING` + `XCLAIM` pair) once it's been
pending for at least that long. Set it too short, and a slow-but-alive
worker's in-progress job gets stolen and processed twice unnecessarily;
too long, and a genuinely crashed worker's job sits unrecovered for a
long time. This experiment's demo used `slow_seconds=5.0` on a
deliberately slow job specifically to open a wide, reliable window to
observe and kill a worker mid-processing without racing a fixed guessed
sleep against real process-startup variance -- which an earlier version
of the demo script tried and got wrong (a flat 0.3s pre-kill sleep fired
before the child process had even finished importing and claimed
anything).

---

**11. Why is Redis Streams a better fit here than a plain Redis list with `LPUSH`/`BRPOP`, and why not just reach for Kafka?**

A plain list has no concept of "claimed but not yet finished" at all --
`BRPOP` removes the item from Redis the instant it's popped, so if that
consumer crashes before finishing, the item is simply gone with no way to
recover it. That would fail this experiment's own worker-crash-recovery
requirement outright. Streams' consumer groups give exactly the missing
piece for free: a claimed-but-unacked entry stays visible in a Pending
Entries List until acked or explicitly reclaimed. Kafka would solve the
same problem, but it brings partitioned multi-broker storage, its own
operational surface, and retention-tiered log semantics that nothing
about this workload actually needs -- a single logical ingestion queue
with consumer-group-based redelivery. This matches a pattern this whole
codebase already follows repeatedly (Redis itself, LangGraph, and a
second MCP-transport layer were all separately evaluated and explicitly
deferred elsewhere in this same repository until a concrete requirement
existed): don't add infrastructure the actual problem doesn't need yet.

---

**12. Is a distributed lock (e.g. Redis `SET NX PX`) always the right way to prevent two workers from racing on the same resource? When would you prefer something else?**

No, and this experiment builds the counterexample directly.
`RacyDocumentIdTable`'s race (modeled on this codebase's own real,
already-fixed `get_or_create_document_id` bug) is fixable with a Redis
lock -- proven -- but it's also fixable with *no lock at all*, just one
atomic check-and-set (`dict.setdefault` under a single lock standing in
for Postgres's real `INSERT ... ON CONFLICT DO NOTHING RETURNING`). The
atomic-write version is strictly better here: it protects every caller
automatically (a lock only protects callers disciplined enough to
remember to acquire it first), and it has no extra network round trips or
TTL-expiry edge cases to reason about (this experiment's own
`RedisDistributedLock` needed a token-checked release specifically to
avoid one caller releasing a lock a different caller had already
re-acquired after the first one's TTL expired). A distributed lock earns
its place instead when the protected operation genuinely spans multiple
non-atomic steps that no single database write can express as one
operation -- e.g. read from one system, call an external API, write the
result to another system -- where there is no equivalent single atomic
primitive available.

---

**13. Your `RedisDistributedLock.release()` checks a token before deleting the key. What bug does that prevent, specifically?**

Without the token check, `release()` would just be an unconditional
`DEL`. Consider: worker A acquires the lock with a 50ms TTL, but takes
longer than 50ms to finish its work; the lock expires and worker B
acquires it (correctly -- the TTL's whole job is to prevent a crashed
holder from wedging the lock forever); worker A finally finishes and
calls `release()`, which would delete the key -- except now that key is
*B's* lock, not A's, so A just released a lock it no longer owned while B
still believes it's holding it. A subsequent worker C could then acquire
the "free" lock while B is still mid-operation, and now B and C are both
inside the critical section at once -- exactly the race the lock existed
to prevent, reintroduced by an unsafe release. The fix: each acquire
stores a random token as the lock's value; release runs a small Lua
script that only deletes the key if its current value still matches the
token *this specific instance* set, so a stale holder's release is
silently a no-op instead of hijacking someone else's lock. Directly
proven in this experiment's own test suite.

---

**14. What's a concrete example, from this experiment, of "process-local state that is actually correct," so I know you're not just pattern-matching "any in-memory state is a bug"?**

`src/rag/observability/metrics.py`'s Prometheus counters. Each `rag-api`
replica keeps its own request/latency/error counts in its own process
memory, and that's exactly right: Prometheus's whole model is "scrape
every replica's own `/metrics` endpoint and sum/aggregate centrally,"
so per-replica counters that never talk to each other are the intended
design, not an oversight. The distinguishing question isn't "is this
state in one process's memory," it's "does the thing being counted need
one single, globally-agreed answer, or is it fine (even correct) for each
replica to only know about its own slice, with aggregation happening
somewhere else." A rate limit needs the former (one global count, or the
limit means nothing); a per-replica request counter needs the latter.

---

**15. If you deployed this to Kubernetes and ran `kubectl scale deployment/ingestion-worker --replicas=3`, what would you expect to observe, and how would you verify it actually worked?**

I'd expect claim throughput on the shared backlog to increase roughly in
proportion to replica count (each new pod is a new, independent consumer
group member claiming disjoint stream entries with zero extra
coordination needed, since Redis Streams already partitions delivery
across group members), each pod's `--name` derived from its own stable
pod identity via the downward API so no two replicas ever collide on a
consumer name, and `active_worker_count()`'s heartbeat-based gauge
climbing to 3 within one heartbeat TTL window. To verify it actually
worked rather than assuming it did from the manifest alone, I'd check:
`kubectl get pods -n rag -l app=ingestion-worker` shows 3 Running pods;
`job_status.py --summary` (or the `ingest_active_workers` Prometheus
gauge) reports 3 active workers, not still 1; submitting a fresh backlog
and timing how long it takes to drain shows a real, measured throughput
improvement over the 1-replica baseline (matching the shape of this
session's own local-process proxy: 1 worker vs. 3 workers on an identical
20-job backlog, 2.8x-4.2x faster across repeated runs); and that no job
was claimed by two consumer-group members simultaneously (every completed
`job_id` appears exactly once as `COMPLETED`, never twice, in the job
store). I would explicitly flag, the same way this experiment's own
README does for Experiments 11-12, that none of this was run against a
real cluster in this session -- the manifests are reviewed and internally
consistent, not smoke-tested.
