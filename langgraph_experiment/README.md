# LangGraph experiment

A focused, isolated learning exercise: build a second, parallel agent
workflow for this repository's RAG/agentic-RAG system using LangGraph,
specifically to find out when LangGraph earns its keep over the existing
custom bounded agent harness (`src/rag/agent/graph.py`, **left completely
unmodified** by this package).

This is **not** a migration. `rag.agent.graph.run_agent` is still what
`POST /agent/query`/`POST /agent/query/stream` call in production. Nothing
under `langgraph_experiment/` is imported by `src/rag/` or `rag.api`
anywhere.

## What this demonstrates

- `StateGraph`, nodes, edges, conditional edges, typed graph state
- A durable checkpointer: **Postgres-backed by default** (SQLite as a
  lighter-weight fallback), not the in-memory default
- `interrupt()` / `Command(resume=...)`: pausing a run mid-flight
- Human-in-the-loop approval as **trusted runtime state**, never an
  LLM-generated claim
- Resuming a paused run **after a real process restart**
- `RetryPolicy` on a node, and inspecting a genuinely failed node's
  checkpoint state
- A worked idempotency story with **three** distinct guarantees, not
  one: workflow durability, business-operation idempotency, and why
  neither one is transaction atomicity, including a real,
  Postgres-backed idempotency ledger proving a crash between a mutation
  executing and the workflow recording it can't double-execute
- Approval expiration, a separate per-workflow timeout, and explicit
  cancellation, all enforced server-side, never trusted from a
  CLI-side check alone
- Checkpoint retention/cleanup as an explicit, operator-run sweep
- An OpenTelemetry span per node and Prometheus workflow/node counters,
  reusing this repo's own observability infrastructure
- A structured, secrets-free audit trail for the whole approval lifecycle

## Dependency isolation

`langgraph>=1.2` pins `langchain-core>=1.4,<2`. This repository's
`pyproject.toml` pins `langchain>=0.3,<0.4` (for `chunkers/
recursive_chunker.py` and `eval/ragas_adapters.py`), which in turn needs
`langchain-core<0.4`. Confirmed directly (`pip install --dry-run
langgraph langgraph-checkpoint-sqlite` inside the project's real `.venv`),
not assumed: the two cannot coexist in one environment.

So this experiment runs in its **own venv**, `.venv-langgraph/` (gitignored,
sibling to the project's `.venv/`). It imports `rag.*` modules directly
(the project is installed editable, `--no-deps`, into that venv) but never
the two modules that import `langchain`: neither ingestion nor RAGAS
evaluation is exercised here, so that's never a problem in practice. See
`__init__.py`'s module docstring for the exact confirmation.

This is itself a real, interview-relevant finding, not a hypothetical: as
of late 2026, LangGraph 1.x's `langchain-core` floor is incompatible with
any project still on the LangChain 0.x line. See Q15 in the interview
section below for the LangGraph/LangChain distinction this actually turns
on.

## Setup

```bash
bash langgraph_experiment/setup_venv.sh      # macOS/Linux/Git-Bash
# or
powershell -File langgraph_experiment/setup_venv.ps1   # native Windows
```

Then activate it (`source .venv-langgraph/Scripts/activate` on
Windows/Git-Bash, `source .venv-langgraph/bin/activate` on macOS/Linux)
before running anything below.

The case-store branch's *business logic* (read/write case tools) needs no
external services. `rag.mcp.business.store` is a self-contained,
in-memory synthetic backend. But `cli/run_demo.py`/`approval_cli.py`
default to the **Postgres-backed checkpointer** (`--backend postgres`)
and the **idempotency ledger** (both needing `make up`'s Postgres
running), since that's the production-shaped default this package's
later, hardened phase settled on. Pass `--backend sqlite --no-ledger`
for a fully offline run, same as the original phase. The read-only RAG
branch additionally needs native Ollama plus an ingested `techfusion` (or
any) dataset.

## Architecture

```
START
  |
classify (deterministic: case-mutation regex, CASE-<id> extraction)
  |
  +-- read_only  -> retrieve -> synthesize_rag -> END
  |                 (real RetrievalPipeline.retrieve() + real LLM)
  |
  +-- case_read  -> select_case_tool -> execute_case_read_tool
  |                 -> evaluate_case_read -> synthesize_case_read -> END
  |                 (rag.mcp.business.store.get_customer_case/
  |                  get_case_status, unmodified)
  |
  +-- case_write -> validate_write_request
                       +-- reject_invalid  -> synthesize_write_result -> END
                       +-- await_approval  -> wait_for_approval  <-- INTERRUPT
                                                (durably paused here)
                                                |
                                       Command(resume=...)
                                                |
                                +-- rejected/denied -> synthesize_write_result -> END
                                +-- execute -> execute_write_action
                                                (rag.mcp.business.store.
                                                 update_case_status, unmodified)
                                                -> synthesize_write_result -> END
```

`graph.py`'s module docstring carries this same diagram next to the
actual `add_edge`/`add_conditional_edges` calls that implement it.

## Directory map

```
langgraph_experiment/
  README.md              this file
  requirements.txt       langgraph + langgraph-checkpoint-sqlite/-postgres
  setup_venv.sh/.ps1      build .venv-langgraph/ and install everything
  state.py                GraphState TypedDict (plain, JSON-safe values)
  identity.py             DemoIdentity -> VerifiedIdentity/AuthorizationContext
  wiring.py               GraphDeps: injected pipeline/llm/case-store fns/ledger
  nodes.py                every node function (+ classify/extraction helpers)
  routing.py              conditional-edge decision functions
  graph.py                build_graph(): all add_node/add_edge wiring
  checkpointer.py         sqlite_checkpointer()/postgres_checkpointer()
  idempotency.py          ActionLedger: the Postgres-backed idempotency ledger
  audit.py                isolated, secrets-free workflow audit-event logger
  observability.py        Prometheus counters + per-node OTel span wrapper
  cli/
    run_demo.py            start a run / inspect a thread's state
    approval_cli.py         list pending approvals / approve / reject / cancel
    two_step_case_demo.py   the CASE-1001 two-mutation scenario, in-process
    compare_harness.py      side-by-side timing + qualitative comparison
    retention_cleanup.py    delete completed threads' checkpoints past a cutoff
  tests/                  see "Testing" below
  data/                   gitignored; checkpoints.sqlite lives here
```

## Scope decisions

Choices made deliberately, each to keep the graph-mechanics teaching goal
uncluttered. Documented here rather than left implicit:

- **Deterministic routing and extraction (`classify`, `select_case_tool`,
  `nodes._extract_case_id`/`_extract_target_status`), no LLM classify
  step.** `rag.agent.graph`'s own custom harness already demonstrates
  LLM-driven structured decisions (`agent/decisions.py`); repeating that
  here would teach nothing new about LangGraph and would make the
  interrupt/checkpoint/resume tests this package exists for flaky on
  JSON-parsing failures instead. `classify` reuses `rag.agent.graph.
  _looks_like_case_mutation_request` directly rather than reimplementing
  that regex.
- **Deterministic, templated synthesis for the case-read and case-write
  branches** (`nodes.synthesize_case_read`/`synthesize_write_result`), not
  a second LLM call. Keeps the entire interrupt/approval/restart demo
  runnable with zero external services. The read-only RAG branch still
  uses a real `LLM`/prompt template, matching production.
- **The write branch always pauses for approval, regardless of whether the
  named transition is actually "sensitive."** `rag.mcp.business.store.
  _SENSITIVE_TRANSITIONS` only requires approval for `resolved -> closed`;
  this graph's `validate_write_request` never inspects that rule, so *any*
  well-formed write request reaches `wait_for_approval`. This is a
  deliberate, more-conservative-than-production choice: it lets the
  interrupt/resume demo run reliably against any case/transition (including
  the prompt spec's own literal "close CASE-1001" example), and makes
  explicit that the graph-level pause is a **UX control layered on top
  of**, never a substitute for, `update_case_status`'s own independent,
  unmodified transition/approval enforcement, which still runs, exactly
  as in production, the moment `execute_write_action` finally calls it.
- **Plain `TypedDict` state (`state.py`), not the real `rag.schemas.
  SearchResult`/`rag.mcp.business.schemas.CustomerCase` pydantic models.**
  Either works with LangGraph's SQLite checkpointer; plain dicts keep
  every checkpoint trivially human-readable (`sqlite3 langgraph_experiment/
  data/checkpoints.sqlite`) and decoupled from `rag`'s own pydantic model
  versions.
- **Case-store tools call `rag.mcp.business.store` directly, in-process --
  not through the real MCP protocol/transport** `rag.agent.mcp_client`
  uses in production. The MCP transport (session handshake, internal
  token minting, ASGI bridging) is a separate, already-covered concern
  (see `docs/architecture.md`'s "MCP integration" section); this
  experiment is about graph orchestration, not re-proving that transport
  layer.
- **One thread = one query's lifecycle**, not a multi-turn conversation.
  Every demo command below starts a fresh thread id for a new query.

## Run commands

`run_demo.py`/`approval_cli.py` default to `--backend postgres` (needs
`make up`). Pass `--backend sqlite --no-ledger` on every command below
for the fully offline original-phase behavior instead.

```bash
# Case-read (needs make up; add --backend sqlite --no-ledger for zero services)
python -m langgraph_experiment.cli.run_demo start "What is the status of CASE-1001?" \
    --tenant tenant_alpha --role tenant_alpha_operator --no-retrieval

# Read-only RAG (needs make up + native Ollama + an ingested dataset)
python -m langgraph_experiment.cli.run_demo start "What is the password rotation policy?" \
    --dataset-id techfusion

# Inspect a thread's checkpointed state (same --backend/--db as the start that created it)
python -m langgraph_experiment.cli.run_demo state <thread_id>

# List every thread currently paused awaiting approval
python -m langgraph_experiment.cli.approval_cli list

# Approve / reject / cancel a paused thread
python -m langgraph_experiment.cli.approval_cli approve <thread_id> \
    --approver-subject alice --approver-roles case_status_approver
python -m langgraph_experiment.cli.approval_cli reject <thread_id> --reason "not yet"
python -m langgraph_experiment.cli.approval_cli cancel <thread_id> --reason "superseded"

# The CASE-1001 two-mutation scenario, run in one process (see the
# limitation note below for why)
python -m langgraph_experiment.cli.two_step_case_demo

# Delete completed threads' checkpoints older than a cutoff (never a
# still-paused thread, regardless of age)
python -m langgraph_experiment.cli.retention_cleanup --older-than-days 7 --dry-run

# Timing + qualitative comparison against the custom harness
python -m langgraph_experiment.cli.compare_harness write-action
python -m langgraph_experiment.cli.compare_harness rag "What is the password rotation policy?" \
    --dataset-id techfusion

# Tests (the idempotency-ledger and write-retry-with-ledger tests need
# `make up`; every other test self-skips cleanly without it)
python -m pytest langgraph_experiment/tests -v
```

## Demo sequence

### 1. Human-in-the-loop, across two real separate processes (CASE-2001)

CASE-2001 seeds as `status="resolved"`, so "close" is the one sensitive
transition (`resolved -> closed`), a clean single-mutation scenario.

```bash
python -m langgraph_experiment.cli.run_demo start "Please close CASE-2001" \
    --thread demo-1 --tenant tenant_beta --role tenant_beta_operator --no-retrieval
# -> status: PAUSED (awaiting approval), prints the interrupt payload

python -m langgraph_experiment.cli.approval_cli list
# -> thread_id=demo-1 case_id=CASE-2001 new_status=closed ...

python -m langgraph_experiment.cli.approval_cli approve demo-1 \
    --approver-subject alice --approver-roles case_status_approver
# -> status: executed / "Case CASE-2001 was updated from 'resolved' to 'closed'."
```

Each of those three commands is a genuinely separate `python -m ...`
process. `approve` reopening the same `.sqlite` file and finding the exact
pending interrupt is the durable-checkpoint story, demonstrated for real,
not simulated.

### 2. Restart/resume, the exact 8-step sequence from the spec

1. `run_demo.py start "..."`: start the workflow.
2. It reaches the approval interrupt (printed as `PAUSED`).
3. The process exits (the command returns; there is no long-running
   server here to "stop"; the whole point is that nothing needs to
   stay running).
4. A brand-new `python -m langgraph_experiment.cli.approval_cli ...`
   invocation starts: a fresh interpreter, fresh import of every
   module, its own new `SqliteSaver` connection.
5. It calls `graph.get_state(config)` and finds the exact same pending
   interrupt the first process left behind.
6. `approve <thread_id> ...`: the human decision.
7. That same call resumes the graph (`Command(resume=...)`).
8. `execute_write_action` runs, `update_case_status` mutates the real
   (in-process, see the limitation note below) case record, and
   `synthesize_write_result` produces the final answer.

`tests/test_restart_resume.py::
test_full_start_stop_restart_approve_resume_complete_sequence` is this
exact sequence, reproduced as an automated test (closing and reopening a
real `SqliteSaver` connection against a `tmp_path` file, not `InMemorySaver`).

### 3. Rejection causes no mutation

```bash
python -m langgraph_experiment.cli.run_demo start "Please close CASE-2001" \
    --thread demo-2 --tenant tenant_beta --role tenant_beta_operator --no-retrieval
python -m langgraph_experiment.cli.approval_cli reject demo-2 --reason "not yet"
# -> status: rejected. CASE-2001's status is untouched.
```

### 4. Duplicate resume is a no-op

Running `approval_cli.py approve demo-1 ...` a second time (after step 1
already completed) prints "has no pending approval; nothing to do" and the
already-recorded outcome, rather than attempting the mutation again. See
"Idempotency and safety" below for the two independent guarantees behind
this.

### 5. The CASE-1001 two-step scenario, and a real limitation found by actually running this

The prompt spec's own example is "close CASE-1001." CASE-1001 seeds as
`status="in_progress"`, and `in_progress -> closed` isn't a valid direct
transition. It needs `in_progress -> resolved` first (not sensitive, but
still pauses in this graph, see "Scope decisions"), then
`resolved -> closed` (the sensitive one). Two separate approved write
actions.

**Discovered while building this demo, not designed in from the start:**
running those two mutations as *separate CLI process invocations* (the
way step 1's single-mutation CASE-2001 demo works perfectly) does not
work for CASE-1001, because `rag.mcp.business.store._SYNTHETIC_CASES` is
an in-memory, per-process Python `dict` with **no persistence of its
own**. That's entirely correct for its real purpose: a synthetic
stand-in backend living inside one continuously-running `rag-api` server
process, never itself needing to survive a restart, but it means a
second `python -m ...` invocation re-imports the module and re-seeds the
original data, silently discarding whatever the first invocation's
in-memory mutation did. The symptom, hit directly while writing this
README: approving "resolve CASE-1001" printed a genuine `executed`
outcome, but the very next `close CASE-1001` attempt (a fresh process)
still saw `previous_status: 'in_progress'` and failed with
`invalid_transition`. The resolve had "worked" and then evaporated.

This is not a LangGraph bug, and it is not a bug in this graph's logic --
it is direct, empirical proof of the "what LangGraph does NOT solve"
point in the interview section below: the SQLite checkpointer genuinely
persists the *workflow's own state* across process restarts (verified in
`test_restart_resume.py`), but that guarantee says nothing about any
external system a node happens to call into. `cli/two_step_case_demo.py`
runs both mutations within one process (four `graph.invoke()` calls
against one `SqliteSaver`, not four separate CLI commands) specifically
so this two-step scenario has a demo that actually works end to end
without needing a real persistent case backend. See that script's
module docstring for the same explanation, next to the code.

## Idempotency and safety

Five scenarios from the spec, and which layer actually prevents a problem:

| Scenario | What actually prevents duplication/corruption |
|---|---|
| `approve <thread>` called twice | **Workflow layer** (`approval_cli._resume`): `graph.get_state(config).interrupts` is empty after the first resume, so the second call never invokes the graph again at all; it just reports the already-recorded outcome. |
| The approval endpoint is retried (e.g. a client-side timeout retry) | Same as above, **plus** the business layer and the idempotency ledger as independent lines of defense. See the next two rows. |
| The action already completed | **Business layer** (`rag.mcp.business.store.update_case_status`): calling it again with the same `(case_id, new_status)` returns the deterministic `already_in_status` outcome and makes no further mutation, *regardless* of whether the workflow-layer guard above was somehow bypassed. `tests/test_idempotent_resume.py::test_business_layer_is_independently_idempotent_even_if_graph_guard_is_bypassed` proves this by calling the business function directly a second time, sidestepping the graph entirely. |
| Process crashes after mutation but before workflow completion is persisted | **The idempotency ledger** (`idempotency.py`'s `ActionLedger`, Postgres-backed, independent of the LangGraph checkpoint) plus the business layer's own natural idempotency, together. LangGraph checkpoints *between* node executions, not mid-node. If a crash happens between `update_case_status` returning and `execute_write_action`'s return value being checkpointed, the workflow resumes at `execute_write_action` again and calls `update_case_status` a second time regardless. `ActionLedger.record_attempt` durably records "an attempt started" *before* calling the mutation, so a resumed/retried attempt can tell "already fully completed" (replay the recorded outcome, never re-call) apart from "started, unknown outcome" (the real crash window; recovers by calling the mutation again, safe here only because it's naturally idempotent). `tests/test_idempotency_ledger.py::test_crash_between_mutation_and_ledger_completion_does_not_double_mutate` reproduces this exact window by hand and proves the case is mutated exactly once in real effect. |
| A `RetryPolicy`-triggered automatic retry of the write node | **The same ledger check, run again on every attempt.** `nodes.make_execute_write_action_node` checks the ledger *inside* the node function, so a `RetryPolicy`-driven re-entry (see `graph.py`'s `_WRITE_ACTION_RETRY_POLICY`) goes through the identical started/completed logic as a manually resumed crash recovery. `tests/test_write_action_retry_policy.py::test_retry_after_the_real_mutation_already_committed_does_not_double_transition` proves a retry that fires *after* the real mutation already committed reports `already_in_status` on the retry, never a fabricated second `executed`. |

**Which guarantees belong to which layer, precisely** (the prompt spec's
three-way distinction):

- **Workflow durability** (LangGraph + its checkpointer) guarantees a
  run's *declared state* and *position in the graph* survive a process
  restart, and that `interrupt()`/`Command(resume=...)` correctly pause
  and resume exactly one pending task per thread. Proven directly in
  `test_restart_resume.py`. Says **nothing** about whether a node's own
  side effect ran zero, one, or more times. That's a separate claim
  entirely.
- **Business-operation idempotency** is the property that calling the
  same logical mutation twice is safe. Two independent layers provide it
  here: `update_case_status`'s own `already_in_status` branch (always
  present, needs no ledger), and `ActionLedger` on top of it (needs
  Postgres, adds the "was an attempt already in flight" check a bare
  natural-idempotency check can't answer on its own. See the crash-row
  above). **Neither the checkpointer nor the ledger *creates* this
  property for an arbitrary mutation.** Both rely on
  `update_case_status` already having it. A genuinely non-idempotent
  side effect (e.g. "charge $10") would need a different recovery
  strategy than "just call it again". See `idempotency.py`'s module
  docstring.
- **Transaction atomicity** is what's conspicuously *absent* here, and
  deliberately so: there is no single atomic transaction spanning "call
  `update_case_status`" and "mark the ledger row complete" (they're two
  separate calls into two different systems (the in-memory case store
  and the Postgres ledger table), with a real crash window between
  them). Postgres gives atomicity *within* one SQL statement; it never
  gives it *across* two separate application-level calls just because
  both happen to eventually write to Postgres. This experiment does not
  paper over that gap. It makes the gap real and testable
  (`test_crash_between_mutation_and_ledger_completion_does_not_double_
  mutate`), then shows why the *outcome* is still safe (business-layer
  idempotency), never claiming the gap itself was closed.

**The checkpointer** (Postgres or SQLite) is what makes workflow
durability real rather than merely in-process. Swap in `InMemorySaver`
and the exact same graph loses all of it on process exit (see
`tests/conftest.py`'s `graph` fixture vs. `checkpointer.
postgres_checkpointer`/`sqlite_checkpointer`).

## Production hardening

A later phase of this experiment (a second round of prompt-spec
requirements, explicitly framed as "for the final implementation")
added the pieces below on top of everything above. Nothing here changes
the original phase's architecture (still the same `classify -> route ->
.../validate/interrupt/execute/synthesize` graph); every addition is
either a new, narrowly-scoped module (`idempotency.py`/`audit.py`/
`observability.py`) or a small extension to an existing node.

- **Postgres-backed checkpointing, preferred over SQLite.**
  `checkpointer.postgres_checkpointer(conn_string)` wraps
  `langgraph-checkpoint-postgres`'s `PostgresSaver`, confirmed to be a
  real, supported, reasonably lightweight package (a 52 KB wheel) by
  installing it and round-tripping a full interrupt/resume/restart cycle
  against this project's own local Postgres before writing any of this
  module, not assumed to work. Its tables (`checkpoints`/
  `checkpoint_blobs`/`checkpoint_writes`/`checkpoint_migrations`) live in
  the same `rag` database/`public` schema this project's own
  `documents`/`chunks`/`feedback` tables already use. Distinctly named,
  no collision, but a real production deployment would more likely use a
  dedicated schema/database; kept simple here since that split is exactly
  the kind of "extra infrastructure for appearance" the prompt spec says
  not to add without a demonstrated need. `sqlite_checkpointer` remains
  available (`--backend sqlite`) as a lighter-weight fallback, and is
  what every automated test in this package still uses internally (a
  `tmp_path` file for restart-resume tests, `InMemorySaver` for
  everything else). No test needs the shared dev database just to prove
  the graph's own logic.
- **The idempotency ledger** (`idempotency.py`'s `ActionLedger`) is
  covered in full above, under "Idempotency and safety". This is the
  module that answers the spec's "prove resume cannot execute the
  mutation twice" requirement directly, with a reproduced crash-window
  test, not just an argument.
- **Approval expiration and a separate per-workflow timeout.**
  `PendingCaseAction.expires_at` (set by `validate_write_request` from
  `GraphDeps.approval_timeout_seconds`, default 15 minutes) and
  `GraphState.workflow_started_at` (set once by `classify`, checked
  against `GraphDeps.max_workflow_duration_seconds`, default 1 hour) are
  two distinct ceilings, checked server-side inside `wait_for_approval`
  on every resume, never trusted from a CLI-side pre-check alone (the
  same posture already applied to `approver_roles`). `approval_cli.py`
  does print a friendly heads-up when it can see a request has expired,
  but still calls the graph regardless, so the *authoritative* refusal
  and its `termination_reason` (`approval_expired`/`workflow_timed_out`)
  always come from the node itself. Workflow timeout is checked first
  (the coarser ceiling), so a request that's both expired and
  workflow-timed-out reports the timeout. See
  `tests/test_approval_expiration_and_cancellation.py`.
- **Explicit cancellation**, distinct from a human rejection --
  `approval_cli.py cancel <thread_id>` resumes with
  `{"decision": "cancel"}`, handled as its own `ApprovalState`/
  `termination_reason` (`cancelled`) rather than folded into `rejected`,
  so an audit trail can tell "a human looked at this and said no" apart
  from "an operator aborted this for an unrelated reason (e.g. the
  request was superseded)."
- **Checkpoint retention/cleanup.** `cli/retention_cleanup.py` is a small,
  explicit, operator-run sweep (not a background job; this experimental
  package has no scheduler, and adding one would itself be unjustified
  infrastructure): deletes a thread's checkpoints via
  `checkpointer.delete_thread(thread_id)` only when that thread has
  **already reached a terminal state** (`get_state(config).next == ()`)
  *and* its most recent checkpoint is older than `--older-than-days`. A
  thread still paused awaiting approval is never touched, no matter how
  old. LangGraph itself has no built-in retention policy (the same way
  a database table has no built-in row-expiry unless something adds
  one), so this script is that something.
- **Max retries**, applied carefully. `execute_case_read_tool`'s
  `RetryPolicy` (from the original phase) retries a read with no side
  effect to worry about. `execute_write_action` now has its own
  `RetryPolicy` too (`graph.py`'s `_WRITE_ACTION_RETRY_POLICY`), but a
  `RetryPolicy` on a *write* node is only safe because the idempotency
  ledger check happens first, inside the node, on every attempt including
  an automatic retry; see the "Idempotency and safety" table's last row
  for the test that proves this specifically, not just asserts it.
- **A safe workflow audit trail.** `audit.py`'s `log_workflow_event`
  emits one structured line per lifecycle transition
  (`write_action_requested`/`approval_granted`/`approval_rejected`/
  `approval_denied_insufficient_role`/`approval_expired`/
  `workflow_cancelled`/`workflow_timed_out`/`write_action_ledger_replay`/
  `write_action_executed`, etc.): IDs, enum values, and counts only,
  the same discipline `rag.audit.log_audit_event` already applies in
  production. Deliberately its own small module, not an extension of
  `rag.audit`'s `AuthEventType`. Adding new event names to that shared,
  production `Literal` would mean editing `src/rag/audit.py` for a
  learning experiment, exactly what this package's isolation exists to
  avoid.
- **An OpenTelemetry span per node.** `observability.traced_node` wraps
  every node registration in `graph.build_graph` with `rag.observability.
  tracing.start_span` (fully reused, not reimplemented; that module has
  no agent-specific code in it). One real correctness bug found and fixed
  while building this: `wait_for_approval`'s `interrupt()` call raises
  `langgraph.errors.GraphInterrupt` internally on every pause (a real
  `Exception` subclass), which a naive `except Exception:` around a node
  call would misclassify as a node *failure* metric on every single
  approval wait. Confirmed directly against `langgraph.errors`'s
  exception hierarchy (`GraphInterrupt` subclasses `GraphBubbleUp`,
  itself an `Exception`), not assumed; `traced_node` catches
  `GraphBubbleUp` separately, before the generic failure branch.
- **Prometheus workflow completion/failure/interrupt counters.**
  `observability.py`'s own dedicated `CollectorRegistry` (mirroring
  `rag.observability.metrics`'s pattern, as its own independent registry
  for the same isolation reason as `audit.py`):
  `langgraph_experiment_workflow_runs_total{route,outcome}`,
  `langgraph_experiment_workflow_interrupted_total`,
  `langgraph_experiment_workflow_failed_total{node}`, and
  `langgraph_experiment_node_latency_seconds{node}`. `record_run_outcome`
  is called once per `graph.invoke()` from each CLI entrypoint, mirroring
  `rag.agent.graph.run_agent`'s own run-level-only metrics observation.
- **No JWT/secrets/chain-of-thought in any checkpoint, ledger row, or
  audit line. True from the original phase, still true here.**
  `GraphState`/`PendingCaseAction` only ever carry plain tenant/role
  claims (never a JWT, see `identity.py`), and the two new additions
  (`operation_id`, `expires_at`) are a UUID and a timestamp. The ledger
  table's own columns (`operation_id`/`thread_id`/`action_type`/
  `case_id`/`new_status`/`outcome`/`previous_status`/timestamps) carry
  nothing beyond what `PendingCaseAction` already exposed. `audit.py`'s
  event fields are the same IDs/enums/counts discipline throughout.

## Comparison with the custom harness

`cli/compare_harness.py write-action` prints this table; `cli/
compare_harness.py rag "<query>"` additionally times both harnesses on a
real read-only RAG query (needs Postgres + Ollama).

| Axis | Custom harness (`rag.agent.graph`) | LangGraph experiment |
|---|---|---|
| Code complexity | ~150 lines for the write-action path incl. one Python `while` loop; everything is inline and directly steppable in a debugger. | ~110 lines across `nodes.py`/`graph.py` for the same path, but understanding a run means reading `graph.py`'s edges *and* every node function; harder to step through live (control flow lives in LangGraph's Pregel runtime, not a visible loop). |
| State visibility | `AgentState` is one pydantic object, inspectable at any breakpoint; no built-in history of prior states within a run. | `GraphState` is checkpointed after every node. `get_state_history()` gives the full sequence of states a run passed through, not just the current one. |
| Branching | Plain `if`/`elif` in Python; trivial to read, but the shape of the graph is not queryable at runtime. | `add_conditional_edges()`. The graph's shape is introspectable via `get_graph()`, which is how this README's architecture diagram was cross-checked. |
| Retry handling | None built in; a tool failure is caught and recorded as a failed `ToolCallRecord`, never re-attempted automatically. | `RetryPolicy` on a node is a one-line, declarative retry with backoff. See `graph.py`'s `_CASE_READ_RETRY_POLICY`. |
| Checkpointing | None. A run lives entirely in one Python call stack/process; nothing survives a crash or restart. | Every node transition is durably checkpointed (SQLite here). A run can be inspected, resumed, or replayed after a full process restart. |
| Pause/resume | Not supported at all. A run either completes or is cancelled (`cancel_event`) and abandoned; there is no mechanism to stop mid-run and continue later. | `interrupt()`/`Command(resume=...)` is the core primitive this package exists to demonstrate. |
| Human approval | Exists in production (`case_approvals`), but must be supplied **before** the run starts (a router-resolved request field). No way to pause an in-flight run to ask. | A run can request approval mid-flight and durably wait, potentially across a process restart, for a human response. |
| Latency | One process, one call stack; no checkpoint-write overhead per step. | Each node transition writes a checkpoint row (SQLite here); measurable but small overhead per step; run `compare_harness.py rag` for an actual number on your machine. |
| Debugging | Standard Python debugger works end to end; a stack trace points straight at the failing line. | `get_state(config).tasks[i].error` surfaces a failed node's exception without a live debugger attached; valuable after the fact or across a process boundary, less immediate than a live `pdb` session. |
| Testability | Tests call `run_agent()` directly with fake pipeline/vectorstore/llm; fully synchronous, no extra test infra. | Tests inject fakes the same way (`GraphDeps`), plus assert on checkpoint state directly (`get_state(config).next`, `.interrupts`). See `tests/test_graph_interrupt_flow.py`. |

**LangGraph is not concluded to be better by default.** For everything
the custom harness already does in production today (bounded
decomposition/tool-selection/evidence-sufficiency loops, pre-authorized
write actions, synchronous request/response over HTTP), the custom
harness is simpler code, easier to debug live, and has zero extra
infrastructure (no checkpoint database, no separate dependency stack).
LangGraph earns its complexity specifically where the custom harness has
a real, structural gap: **pausing an in-flight run to wait on something
external** (a human decision, potentially minutes or hours away) **and
surviving a process restart while paused.** That gap doesn't exist in
today's production system (approvals are always pre-authorized before a
run starts), which is exactly why the custom harness not needing
LangGraph today is reasonable, not merely inertia.

## Testing

Most tests need no Postgres/Ollama at all. See "Scope decisions" above.
`tests/conftest.py`'s `_restore_synthetic_cases` autouse fixture
snapshots/restores `rag.mcp.business.store`'s shared in-memory case dict
around every test, the same pattern the main repo's own `tests/unit/
test_mcp_business_case_actions.py` already uses for the same module. The
three idempotency-ledger/write-retry-with-ledger tests need real Postgres
(`require_postgres`/`postgres_deps`, self-skipping cleanly without it,
mirroring `tests/integration/conftest.py`'s identical-purpose fixture in
the main test suite). Run `make up` first to include them.

```bash
python -m pytest langgraph_experiment/tests -v
```

| File | Proves |
|---|---|
| `test_routing.py` | `classify`/extraction/routing are pure, deterministic functions, no graph invocation. |
| `test_graph_read_only_path.py` | The normal read-only RAG path, including the no-evidence and no-pipeline edge cases (fake pipeline/LLM, see "mocking narrowest point" in `conftest.py`). |
| `test_graph_case_read_path.py` | Authorized/unauthorized/nonexistent case reads, against the real `rag.mcp.business.store` data. |
| `test_graph_interrupt_flow.py` | Reaching the approval interrupt, approval executing the mutation, rejection causing no mutation, an approval claim without a real role being denied, and a malformed write request never reaching the interrupt at all. |
| `test_idempotent_resume.py` | Duplicate resume is a workflow-layer no-op; the business layer is independently idempotent even if that guard is bypassed. |
| `test_restart_resume.py` | A real `SqliteSaver` file, closed and reopened (two independent connections, simulating a process restart), reloads the exact paused state and completes, including the exact 8-step sequence from the spec. |
| `test_authorization_server_side.py` | A resume payload with no approver roles is denied; an unauthorized write request still pauses but executes as denied (never leaking whether the case exists); cross-tenant support-role reads work exactly like the document-level ACL rule they mirror. |
| `test_failed_node_state.py` | `RetryPolicy` absorbing a transient failure within budget; a persistent failure propagating with an inspectable `tasks[i].error`; and resuming from that exact failed node once the underlying dependency is fixed. |
| `test_idempotency_ledger.py` | *(Postgres)* A fresh operation executes once; an already-completed operation is replayed without re-mutating; the exact crash-between-mutation-and-completion window is reproduced by hand and proven safe; a full graph run records a completed ledger row. |
| `test_approval_expiration_and_cancellation.py` | Approval expiration and the separate per-workflow timeout are enforced server-side (and the timeout takes precedence); a well-formed approval within both windows still executes; explicit cancellation terminates without mutation; a stale cancel reports expiration, not cancellation. |
| `test_write_action_retry_policy.py` | `execute_write_action`'s own `RetryPolicy` recovers a transient failure; *(Postgres)* a retry that fires after the real mutation already committed reports `already_in_status` on the retry, never a fabricated second `executed`. |

## Interview learning output

### Fifteen concepts

1. **What LangGraph solves.** Durable, inspectable, controllable
   multi-step orchestration: checkpointed state that survives a process
   restart, mid-flight pause/resume on an external event (a human
   decision), declarative retries, and a graph topology you can query
   independently of any node's code. A plain `while`/`if` loop (the custom
   harness) can do decomposition/tool-use/branching too. What it
   structurally cannot do is survive a crash mid-run or pause indefinitely
   waiting on something outside the process.
2. **`StateGraph` vs ordinary Python control flow.** `StateGraph` declares
   nodes and edges as data (an object you can inspect, visualize, and
   compile); the LangGraph runtime (Pregel) drives execution and
   checkpoints between every node. A `while`/`if` loop's "state" is just
   local variables that vanish the moment the process exits or an
   exception unwinds the stack.
3. **Node vs edge.** A node is a unit of work: a function `(state) ->
   partial_state_update`. An edge is the graph's declared transition from
   one node to another. Nodes never call each other directly; only edges
   connect them, so the connection topology is inspectable independent of
   any node's implementation.
4. **Conditional edge.** An edge whose destination is computed at runtime
   from state (`add_conditional_edges(source, decision_fn, path_map)`) --
   the `StateGraph`-native way to branch, as opposed to an `if` embedded
   inside a node function.
5. **Checkpoint.** A durable snapshot of a run's full state plus "what
   node(s) run next," written after every node completes (or pauses).
   This experiment uses `SqliteSaver`; a production LangGraph deployment
   typically uses a Postgres-backed saver instead.
6. **Thread/run ID.** The checkpointer's primary key for one run's whole
   checkpoint history (`config["configurable"]["thread_id"]`). One run is
   one thread in this experiment, but LangGraph also supports addressing
   an individual checkpoint within that history (`get_state_history()`),
   which is how "time-travel" replay/debugging works.
7. **Interrupt.** `interrupt(payload)` pauses the current node exactly at
   that call, surfaces `payload` to the caller (a `"__interrupt__"` key in
   `.invoke()`'s result, and `get_state(config).interrupts`), and durably
   checkpoints the pause. A real, easy-to-miss nuance verified directly
   against the installed `langgraph==1.2.11` while building this
   experiment: on resume, LangGraph **replays the node function from its
   start**, not from the `interrupt()` line. Code before that call runs
   again every time the node is (re-)entered. `nodes.wait_for_approval`
   deliberately has no side effects before its `interrupt()` call for
   exactly this reason.
8. **Resume.** Supplying a value for a paused interrupt via
   `Command(resume=value)` passed to `.invoke()`; the paused node's
   `interrupt()` call returns that value and execution continues from
   there.
9. **Human-in-the-loop.** Using interrupt/resume specifically so a human
   can make a real decision mid-workflow, with the workflow durably
   waiting, with no polling loop and no held-open connection, until that
   decision arrives, even across a process restart.
10. **Durable execution.** The property that a workflow's progress is not
    lost if the process running it dies. Achieved here by checkpointing
    every node transition to a real file (SQLite), verified directly by
    closing and reopening a `SqliteSaver` connection mid-run
    (`test_restart_resume.py`).
11. **What LangGraph does NOT solve.** Decision quality (still the
    prompt/model's job); authorization or business rules (a resume
    payload is just a value, see Q17 below); and any persistence for
    state a workflow's *nodes* call out to but never put into the
    checkpointed `GraphState` itself. This experiment hit that third point
    empirically, not hypothetically. See the "real limitation, found by
    actually running this" section above.
12. **Why business actions still need idempotency.** A checkpoint proves
    the *workflow's* progress, not that a given external call executed
    exactly once. A crash between a side effect completing and its
    result being checkpointed means the workflow can retry that step on
    restart. The real safety net is the business action's own idempotent
    outcome (`already_in_status`), which stays safe regardless of what the
    workflow layer does.
13. **Why the current custom harness is reasonable today.** It doesn't
    need pause/resume (`case_approvals` are resolved before a run starts,
    never mid-flight), doesn't need cross-process durability (it's a
    synchronous, in-request FastAPI handler), and a plain Python loop is
    dramatically easier to read, debug live, and test for that shape of
    problem. See the comparison table above.
14. **When to migrate a production agent to LangGraph.** When you actually
    need one or more of: multi-step workflows that must survive a process
    restart/crash; mid-flight human approval (rather than pre-authorized);
    long-running workflows (minutes to days) that shouldn't hold a
    connection or thread open the whole time; or replaying/auditing a
    workflow's exact state history. Not "because it's more modern". See
    the honest costs in the comparison table.
15. **LangGraph vs LangChain.** LangChain is a library of components
    (prompt templates, document loaders, chains, retrievers) for building
    LLM applications. LangGraph, built by the same team, is a lower-level
    graph/state-machine orchestration runtime for controllable, stateful
    workflows, with its own checkpointing and interrupt primitives.
    LangGraph depends on `langchain-core` (the shared low-level types:
    messages, runnables) but not on the full `langchain` package's
    chains/loaders. This experiment imports `langgraph` without ever
    importing `langchain` itself. This project's actual dependency
    conflict (see "Dependency isolation" above) is specifically
    `langchain-core` version skew, not a conceptual clash between the two
    libraries.

### Twenty likely interview questions

1. **Q: What does `interrupt()` actually pause?**
   A: The current node's execution, at the exact line `interrupt()` is
   called. The whole graph run stops advancing until `Command(resume=...)`
   is supplied for that same thread.
2. **Q: If a node calls `interrupt()` twice, what happens on resume?**
   A: LangGraph replays the node function from its start; the first
   `interrupt()` call, once already resumed, returns its stored value
   immediately, execution continues to the second `interrupt()` call, and
   that one pauses in turn. Sequential interrupts in one node each get
   their own resume value, in order.
3. **Q: What if a node has side effects before `interrupt()`?**
   A: They re-run every time the node is (re-)entered, since LangGraph
   replays a node from its beginning on resume. Put irreversible work
   after the interrupt, not before it. See concept 7 above.
4. **Q: How does LangGraph know which interrupt a resume value is for?**
   A: `Command(resume=value)` with a bare value resumes "the next
   interrupt" on that thread; for multiple concurrent interrupts (e.g.
   parallel branches), `resume` also accepts a mapping of interrupt id to
   value.
5. **Q: Where is checkpoint state actually stored?**
   A: Wherever the checkpointer implementation writes it --
   `InMemorySaver` (process memory, gone on exit), `SqliteSaver` (a local
   file, this experiment), or a Postgres-backed saver in a typical
   production LangGraph deployment.
6. **Q: What's actually inside one checkpoint?**
   A: The full graph state values at that point, which node(s) are
   "next," any pending interrupt payloads, and metadata (step number,
   timestamp).
7. **Q: Does a checkpoint capture the Python call stack?**
   A: No. Only the declared `GraphState` values. A local variable inside
   a node function that isn't returned as part of the state update is
   gone once that node call ends.
8. **Q: What happens if a node raises an uncaught exception?**
   A: It propagates out of `.invoke()`. The checkpoint from before that
   node started is still there; `get_state(config).next` names the failed
   node, and `.tasks[i].error` carries the exception's string
   representation, resumable once the underlying cause is fixed, proven
   directly in `test_failed_node_state.py`.
9. **Q: How does `RetryPolicy` interact with an exception it shouldn't
   retry?**
   A: `retry_on` (a type, tuple of types, or predicate) filters which
   exceptions get retried at all; anything else propagates immediately
   after the first failure.
10. **Q: Can a LangGraph node function be non-deterministic?**
    A: Yes in general, but a node with side effects before an
    `interrupt()` call, or one relied on for exactly-once behavior, needs
    the same idempotency discipline as any retry-prone distributed-systems
    code. See concept 12.
11. **Q: What's the difference between `thread_id` and a checkpoint id?**
    A: `thread_id` identifies one run's whole history (many checkpoints);
    each individual checkpoint additionally has its own id, letting you
    address or replay from a specific point in that history.
12. **Q: Can a graph fork/replay from an earlier checkpoint?**
    A: Yes. `get_state_history(config)` returns every prior checkpoint's
    snapshot; invoking against a config pinned to an earlier checkpoint id
    continues from there instead of the latest one. Not exercised in this
    experiment, but is how LangGraph's "time-travel" debugging works.
13. **Q: Why plain `TypedDict`s for state instead of pydantic models?**
    A: Either works with LangGraph's checkpointer; this experiment chose
    plain dicts so every checkpoint is trivially human-readable JSON and
    stays decoupled from `rag`'s own pydantic model versions, a
    pedagogical/simplicity choice, not a LangGraph requirement.
14. **Q: What does `add_conditional_edges`'s `path_map` actually do?**
    A: Maps the decision function's return value to a target node name.
    This experiment always passes one explicitly, for clarity.
15. **Q: Does compiling the same `StateGraph` object twice give
    independent graphs?**
    A: Yes. Each `.compile(checkpointer=...)` call returns its own
    `CompiledStateGraph`; this experiment's tests recompile against a
    fresh `InMemorySaver` per test for isolation.
16. **Q: Why does this graph's write-action branch always pause, even for
    a non-sensitive transition?**
    A: A deliberate simplification (see "Scope decisions"). Production's
    own `rag.mcp.business.store` already independently enforces which
    transitions actually require approval; this graph's pause is a UX
    layer on top, not a substitute for that check.
17. **Q: What stops a forged resume payload from claiming approval?**
    A: Nothing about `interrupt`/`resume` itself. It's just a value.
    `wait_for_approval` re-checks the resume payload's `approver_roles`
    against config's `approval_roles` before ever treating it as approved
    (`test_graph_interrupt_flow.py::
    test_approval_claim_without_approver_role_is_denied`). The trust
    boundary is application code, not a LangGraph feature.
18. **Q: Why isn't duplicate-resume safety just "LangGraph handles it"?**
    A: Two independent guarantees, not one. The workflow layer's own
    state (`get_state(config).interrupts` being empty) prevents a second
    `Command(resume=...)` call from re-entering a completed run's approval
    step at all, but if that check were skipped or raced, the business
    layer's `already_in_status` outcome is what actually prevents a
    duplicate mutation. See the "Idempotency and safety" table above.
19. **Q: What's the actual overhead of checkpointing per step?**
    A: A row written to the checkpointer (SQLite here) per completed
    node, small but nonzero versus the custom harness's zero-checkpoint
    in-process loop; run `compare_harness.py rag` for a measured number.
20. **Q: When would you deliberately not reach for LangGraph?**
    A: A synchronous, single-request-lifetime workflow with no mid-flight
    pause requirement and no need for cross-process durability or
    replayable state history, exactly what `rag.agent.graph.run_agent`
    already is, and does well, with far less code and easier live
    debugging.
21. **Q: Why does an idempotency ledger exist at all if `update_case_status`
    is already idempotent on its own?**
    A: `already_in_status` alone can't distinguish "never attempted" from
    "attempted, crashed mid-flight, unknown outcome". Both look
    identical from outside the business layer. The ledger's `started_at`/
    `completed_at` distinction adds exactly that missing information,
    letting a recovery path choose "replay the recorded outcome" over
    "call the mutation again" once it actually knows the difference.
    Without a naturally idempotent mutation underneath it, though, the
    ledger alone still wouldn't be enough. See concept 12 and Q22.
22. **Q: What would you have to do differently if the mutation weren't
    naturally idempotent (e.g. "charge $10")?**
    A: The "started, not completed" recovery case can no longer safely
    just call the mutation again. See `idempotency.py`'s module
    docstring. It would need to query the external system for the real
    outcome of that specific `operation_id` (if the system supports
    idempotency keys itself, which many payment APIs do), or refuse to
    proceed automatically and require manual reconciliation. This
    experiment's ledger design leans entirely on `update_case_status`
    already having the easier property; it doesn't manufacture safety
    for a mutation that lacks it.
23. **Q: Why check workflow timeout before approval expiration, and not
    the other way around?**
    A: An arbitrary, but principled, choice: `max_workflow_duration_
    seconds` is the coarser, "something is systemically wrong" ceiling
    (an operator-facing SLO), while `expires_at` is a narrower,
    per-request UX window. Reporting the systemic reason first is more
    actionable for an operator reading the audit trail than reporting
    "this one narrow window closed," when in fact the whole workflow is
    the thing that's stuck.
24. **Q: Why is `execute_write_action`'s `RetryPolicy` unsafe on its own,
    without the ledger?**
    A: `RetryPolicy` retries the whole node function from its start on a
    matching exception. If the underlying mutation call had already
    taken effect before the exception was raised (e.g. a response
    genuinely lost after the server-side change committed), a bare retry
    with no memory of that would call the mutation again with no way to
    know it already happened. The ledger's `mark_started` row, written
    before the mutation call, is what lets a retried attempt recognize
    that case instead of retrying blind.
25. **Q: Why does `traced_node` need to special-case `GraphBubbleUp`
    instead of just catching every exception?**
    A: `interrupt()` pausing a node is implemented internally as raising
    `langgraph.errors.GraphInterrupt` (a real `Exception` subclass) to
    unwind the stack, a normal, expected control-flow signal on every
    single approval wait, not a node failure. A generic `except
    Exception:` around a node call would silently mislabel every pause as
    a failure in `WORKFLOW_FAILED_TOTAL`; the fix is to catch
    `GraphBubbleUp` (its parent class) first and re-raise it untouched,
    before the generic failure-counting branch ever sees it.
