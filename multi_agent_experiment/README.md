# Multi-agent LangGraph experiment

A focused, isolated learning exercise: build a small **multi**-agent
LangGraph workflow: a Coordinator/Router plus two genuinely distinct
specialist agents (different tools, different responsibilities), on top
of this repository's RAG/agentic-RAG system, and compare it honestly
against the two agents that already exist here:

- **A.** the production custom harness (`rag.agent.graph.run_agent`)
- **B.** the single-agent LangGraph experiment (`langgraph_experiment/`)
- **C.** this package, the multi-agent LangGraph experiment

Neither A nor B is modified by this package. Nothing under
`multi_agent_experiment/` is imported by `src/rag/`, `rag.api`, or
`langgraph_experiment/` anywhere. This package imports *them*, not the
other way around.

## What this demonstrates

`StateGraph` with genuinely distinct nodes, `Send`-based dynamic parallel
fan-out, a fan-in join node (and a real bug found and fixed in getting
that join correct, see "Issues found" below), bounded retry loops that
route back through a coordinator, structural tool isolation between
specialists (verified statically, not just asserted), reuse of an
already-tested business-write branch unmodified, and a deterministic,
zero-external-service evaluation harness.

## Directory map

```
multi_agent_experiment/
  README.md                this file
  __init__.py               package docstring: scope, non-goals, venv reuse
  limits.py                 bounded-execution constants (retry/round caps)
  state.py                  MultiAgentState TypedDict + reducer discipline
  knowledge_agent.py         the Knowledge Agent (one specialist, one tool)
  business_agent.py          the Business Agent (reuses langgraph_experiment.nodes unmodified)
  orchestrator.py            coordinator / merge / evidence_critic / final_synthesis
  routing.py                 conditional-edge decision functions (Send-based fan-out)
  graph.py                   build_graph(): wires every node/edge, the topology diagram lives here
  checkpointer.py            sqlite_checkpointer(), pointed at this package's own data/
  cli/
    run_demo.py               start a run / inspect a thread's state
    approval_cli.py            list pending approvals / approve / reject
    scenario_walkthrough.py    all 5 required scenarios, zero external services, one command
    compare_harness.py         A vs B vs C timing + the full qualitative comparison table
  eval/
    multi_agent_gold.jsonl     12-row gold set (see "Evaluation results" below)
    run_multi_agent_eval.py    deterministic evaluation harness, zero external services
  tests/                     see "Testing" below (28 tests, all passing, zero external services)
  data/                      gitignored; checkpoints.sqlite lives here
```

## Files added/changed

**Changed:** nothing. `src/rag/`, `langgraph_experiment/`, and every
other existing file (including the repo-root `.gitignore`, whose
`.venv-langgraph/` entry predates this work) are byte-identical to
before this work. Verified via `git diff` before writing this
section, not assumed.

**Added:** exactly the 30 files under `multi_agent_experiment/` listed in
the directory map above: 9 core modules (incl. `__init__.py`), 5 CLI
scripts (incl. `__init__.py`), 3 eval files (incl. `__init__.py` and the
gold-set JSONL), 9 test files (incl. `__init__.py`), plus `README.md`,
`.gitignore`, and `data/.gitkeep`. No
change to `pyproject.toml`, `requirements*.txt`, or any file
`langgraph_experiment/setup_venv.sh`/`.ps1` installs. This package reuses
that venv and those dependencies exactly as they already exist.

## Reuse map (what is new code vs. reused)

- **Reused, unmodified:** the entire business-write branch
  (`select_case_tool`, `make_execute_case_read_node`,
  `validate_write_request`, `make_wait_for_approval_node`,
  `make_execute_write_action_node`, `synthesize_write_result`, and the
  `route_after_validate`/`route_after_approval` conditional edges) is
  imported directly from `langgraph_experiment.nodes`/`.routing` and
  wired into this graph as-is. `GraphDeps`/`build_default_deps`
  (`langgraph_experiment.wiring`) is reused wholesale; no second
  dependency-injection type for this package. `rag.mcp.business.store`
  (the actual authorization/transition/approval enforcement) and
  `rag.retrieval.pipeline.RetrievalPipeline` (the actual retrieval
  authorization/freshness/redaction enforcement) are untouched, exactly
  as in production and in B.
- **New:** the Coordinator/router (`orchestrator.coordinator`), the
  Knowledge Agent (`knowledge_agent.make_knowledge_agent_node`), the
  `merge`/`evidence_critic`/`final_synthesis` orchestration nodes, and
  the `Send`-based routing (`routing.route_after_coordinator`). This is
  the actual multi-agent topology; everything else is composition.

## Architecture diagram

```
                              ┌─────────────┐
                    ┌────────▶│   Coordinator│  (deterministic router,
                    │         └──────┬──────┘   round 0: classify; round ≥1: retry-adjust)
                    │                │
        ┌───────────┼────────────────┼───────────────────┬──────────────┐
        │            business mutation                    │              │
        │            (bare string, no Send)                │              │
        │                │                                 │              │
        │        ┌───────▼────────┐                         │              │
        │        │ Business Agent │  (write path, reused    │              │
        │        │  write branch  │   from B unmodified)     │              │
        │        └───────┬────────┘                         │              │
        │                │ interrupt() pauses here           │              │
        │           (human approves/rejects out of band)     │              │
        │                │                                    knowledge     business
        │                ▼                                     only         read only
        │              END                                  (Send x1)     (Send x1)
        │                                                      │              │
        │         ┌───────────────── mixed (Send x2, PARALLEL) ┼──────────────┤
        │         │                                            │              │
        │         ▼                                            ▼              ▼
        │   ┌─────────────┐                             ┌─────────────┐┌─────────────┐
        │   │  Knowledge  │                             │  Knowledge  ││  Business   │
        │   │    Agent    │                             │    Agent    ││    Agent    │
        │   │ (search_kb) │                             │ (search_kb) ││ (read path) │
        │   └──────┬──────┘                             └──────┬──────┘└──────┬──────┘
        │          │                                           │              │
        │          └─────────────┬─────────────────────────────┴──────────────┘
        │                        ▼
        │                 ┌─────────────┐   defer=True: waits for every
        │                 │    merge    │   pending branch, regardless
        │                 └──────┬──────┘   of branch length (see ISSUES.md)
        │                        ▼
        │                 ┌─────────────┐
        │                 │  Evidence   │───"retry" (knowledge only, bounded)──┐
        │                 │   Critic    │                                     │
        │                 └──────┬──────┘                                     │
        │                        │ "proceed"                                  │
        │                        ▼                                           │
        │                 ┌─────────────┐                                    │
        │                 │   Final     │                                    │
        │                 │ Synthesis   │                                    │
        │                 └──────┬──────┘                                    │
        │                        ▼                                           │
        │                      END                                          │
        └───────────────────────────────────────────────────────────────────┘
                        (back to Coordinator)
```

`graph.py`'s own module docstring carries the exact, narrower ASCII
diagram next to the real `add_node`/`add_edge`/`add_conditional_edges`
calls that implement it. This diagram is the conceptual picture;
`graph.py` is the literal one.

## Distinct agents, distinct tools, distinct responsibilities

Per this experiment's own constraint ("do not create multiple agents
merely by giving different prompts to identical capabilities"):

| | Coordinator (`orchestrator.py`) | Knowledge Agent (`knowledge_agent.py`) | Business Agent (`business_agent.py`) |
|---|---|---|---|
| Responsibility | Classify the query, decide which specialist(s) are needed, merge results, trigger synthesis. Never performs a business mutation itself. | Retrieve authorized knowledge-base evidence. | Retrieve/mutate customer-case state through the existing, unmodified MCP business path. |
| Allowed tools | None (reads state written by specialists; classifies via a reused regex, never a tool call). | `search_knowledge_base` (`RetrievalPipeline.retrieve()`) only. | `get_customer_case`/`get_case_status`/`update_case_status` (`rag.mcp.business.store`) only. |
| Forbidden by construction | Any tool call at all. | `rag.mcp.business.*`, never imported (verified statically, see "Tool isolation" below). | `RetrievalPipeline`/`rag.agent.tools`, never imported. |
| Failure containment | Bounded retry decision (`evidence_critic`), bounded round count (`limits.MAX_COORDINATOR_ROUNDS`). | Catches its own tool exception, reports `knowledge_status="failed"`, never crashes the run. | Deterministic transition table; `update_case_status` returns a typed outcome, never raises for a business-rule denial. |

An optional fourth role, an Evidence Critic, *was* added. See
"Evidence Critic: why it earns its place" below for why this one was
judged to add real value rather than cosmetic multi-agent flavor.

## Shared graph state vs. specialist-private concerns vs. trusted server-side context

Three distinct tiers, matching `state.py`'s own module docstring:

1. **Shared workflow state** (`MultiAgentState`): plain, JSON-safe values
   : `original_query`, routing flags, each specialist's own
   `*_status`/`*_evidence`/`*_result` fields (field-owned by exactly one
   specialist, so parallel branches never write-race), `citations`,
   `critic_notes`, `termination_reason`. This is what's checkpointed and
   what a trace reader inspects.
2. **Specialist-local reasoning**: a specialist's own decision of *how*
   to call its one tool (e.g. `knowledge_agent`'s retry-aware `top_k`)
   lives inside that specialist's node function, not exposed as a
   separate state field unless a sibling node genuinely needs it.
3. **Trusted server-side security context**: `AuthorizationContext`
   (tenant/roles, resolved inside `RetrievalPipeline`/`rag.mcp.business.
   store`, never in `MultiAgentState`), the internal MCP service token
   production mints (not used by this experiment at all, see "Business-
   action safety" below), and case-mutation approval (`pending_action`'s
   `approval_state`, only ever set by `business_agent`'s reused,
   role-checking `wait_for_approval` node, never by a tool-argument or an
   LLM-generated claim).

**Never** placed in shared, prompt-visible state: JWTs, MCP internal
tokens, secrets, raw `AuthorizationContext` objects, hidden chain-of-
thought, or unredacted raw tool output. `knowledge_agent`'s evidence is
already sanitized *inside* `RetrievalPipeline.retrieve()` before this
graph ever sees it (field redaction, injection flagging, unchanged,
inherited for free), and `business_agent`'s case data is already the
public, ACL-filtered shape `rag.mcp.business.store.to_public()` returns.

## Required scenarios (README section 7 of the spec)

All five are runnable in one command, with **zero external services**:

```bash
python -m multi_agent_experiment.cli.scenario_walkthrough
```

| Case | Query | Expected | Verified by |
|---|---|---|---|
| 1: pure knowledge | "What is the current API timeout policy?" | Coordinator → Knowledge Agent only; Business Agent never runs. | `tests/test_case1_knowledge_only.py` |
| 2: pure business | "What is the status of CASE-1001?" | Coordinator → Business Agent only; Knowledge Agent never runs. | `tests/test_case2_business_read_only.py` |
| 3: mixed | "According to the support policy, what should happen next for CASE-1001?" | Both specialists dispatched **in the same LangGraph superstep** (proven via `stream_mode="debug"`, not just eventually-both-ran), merged, one grounded answer citing both sources. | `tests/test_case3_mixed_parallel.py` |
| 4: business mutation | "Please close CASE-2001" | Coordinator → Business Agent → the existing MCP write-action path (`interrupt()` pause, real role-checked approval, no invented approval). | `tests/test_case4_business_mutation.py` |
| 5: specialist failure | Knowledge Agent's `retrieve()` raises (simulated timeout); a business read is denied/not-found | Coordinator never fabricates: one bounded retry, then an honest "insufficient evidence" answer with a recorded reason. | `tests/test_case5_specialist_failures.py`, `test_case2_business_read_only.py`'s second test |

Example trace (real output from `scenario_walkthrough.py`, reproducible
exactly since it uses a small, deterministic synthetic knowledge base;
see that script's own docstring for why):

```
======================================================================
CASE 3: mixed question (parallel Knowledge + Business)
======================================================================
needs_knowledge=True  needs_business=True
selected_specialists=['knowledge', 'business']
termination_reason=synthesized
answer: [synthesized answer grounded in]: query=According to the support policy, ...
citations: [{'chunk_id': 'kb-2', 'source': 'docs/support-policy.md', ...},
            {'case_id': 'CASE-1001', 'source': 'case:CASE-1001', ...}]

======================================================================
CASE 5: Knowledge Agent fails -> bounded retry -> no fabrication
======================================================================
termination_reason=max_rounds
answer: I don't have enough authorized, retrieved evidence to answer this question confidently.
critic_notes: ["knowledge specialist returned 'failed' on round 1; requesting one bounded retry
               with a wider top_k (5 -> 10).",
              "knowledge specialist still 'failed' after the retry budget was exhausted;
               proceeding without fabricating an answer."]
```

## Parallel execution

For the "mixed" route, both specialists are dispatched via `Send` in the
same conditional-edge call, so they run as genuinely parallel LangGraph
tasks within one superstep. Confirmed with `stream_mode="debug"`, not
assumed (`tests/test_case3_mixed_parallel.py`). Sequential dispatch was
not built as a second, separate graph variant to time against (that would
duplicate the whole topology for one number); instead
`cli/compare_harness.py mixed` measures the real parallel run, then
computes a **sequential estimate** as the sum of two separately-measured,
independently-run solo calls (a knowledge-only phrasing and a
business-only phrasing of the same underlying case), an honest estimate
built from two real numbers, not a guess, and the command says so
explicitly if it can't compute one.

**Conflicting state updates**: they can't happen here by construction.
`knowledge_agent` only ever writes `knowledge_*`-prefixed keys;
`business_agent`'s reused nodes only ever write `case_*`/`business_*`-
prefixed keys. No two parallel branches write the same `MultiAgentState`
key in the same superstep, so no LangGraph reducer conflict is possible
for those fields. The two fields that genuinely can be written more than
once in a run's lifetime (`knowledge_tool_calls`/`business_tool_calls`,
across a critic-requested retry) use an explicit `Annotated[...,
operator.add]` reducer so their contributions accumulate rather than
last-write-wins; `citations`/`tool_call_log` are deliberately *not*
reducers. `merge` recomputes both from scratch on every invocation, so
a retry's fresher evidence replaces stale evidence rather than being
unioned with it. See `state.py`'s module docstring for the full
per-field reasoning.

## Tool isolation

Verified **statically**, not just by convention: `tests/
test_tool_isolation.py` parses `knowledge_agent.py`/`business_agent.py`/
`orchestrator.py`'s actual `import`/`from` statements via `ast` and
asserts the forbidden modules never appear. This is stronger than a
runtime permission check. A specialist cannot call the other's tool not
because something forbids it at request time, but because the Python
code that could do so was never written in that file. Why this helps:

- **Least privilege**: a bug in `knowledge_agent.py` cannot, even in
  principle, reach `update_case_status`. There is no import path to it.
- **Simpler reasoning**: reviewing "can the Knowledge Agent mutate
  business state?" is answerable by reading one file's import block, not
  by tracing every call site.
- **Easier testing**: `knowledge_agent.py` is testable with only a fake
  `RetrievalPipeline`; `business_agent.py` needs no retrieval fixture at
  all (its own tests use the real, self-contained `rag.mcp.business.
  store`).
- **Failure containment**: a Knowledge Agent failure (`knowledge_status=
  "failed"`) cannot corrupt or block the Business Agent's independent
  branch, and vice versa. Proven directly by `test_case5_specialist_
  failures.py`'s business branch continuing to work while a *different*
  test's knowledge branch is made to fail.
- **Clearer observability**: `tool_call_log`/`critic_notes` are
  attributable to a named specialist, not an undifferentiated agent.

**Agent-level tool isolation does not replace backend authorization.**
Even if `knowledge_agent.py` somehow *did* import `rag.mcp.business.
store` (it doesn't), `update_case_status` still independently enforces
tenant/role authorization, the transition table, and the approval
requirement. Those checks have no notion of "which LangGraph node called
me." This experiment's isolation is a defense-in-depth, blast-radius
control on top of an authorization boundary that would hold regardless,
not a substitute for it. See "Security analysis" below.

## Evidence Critic: why it earns its place

Per the spec's own instruction not to add a Critic merely to look more
multi-agent, `orchestrator.evidence_critic` was scoped to exactly two
mechanical responsibilities that a run genuinely needs and that no other
node already provides:

1. **The one bounded retry decision.** If the Knowledge Agent came back
   empty/failed and the retry budget allows it, request exactly one
   retry with a widened `top_k`, routed back through the Coordinator
   (never a direct retry loop bypassing the round-count bound). The
   Business Agent is deliberately never retried. Its read is a
   deterministic lookup, so a second identical call changes nothing; the
   critic's own note explains this reasoning explicitly in the trace
   rather than silently skipping it.
2. **A no-fabrication guard**, redundant-by-design with `final_synthesis`'s
   own independent zero-evidence check (matching this codebase's
   established "a guarantee that depends on one layer is not a
   guarantee" pattern. See CLAUDE.md's field-redaction-marker
   precedent).

What it deliberately does **not** do: semantic conflict detection between
knowledge and business evidence. That was considered and left out. A
real implementation would need either an LLM call (reintroducing the
flakiness this experiment's deterministic design avoids) or a fragile
keyword heuristic prone to false positives/negatives on exactly the kind
of nuanced claim a conflict check exists to catch. Documented here as an
honest scope decision, not hidden.

## Business-action safety (unchanged from production)

Every one of these was true before this experiment existed and remains
true because `business_agent.py` reuses production's own enforcement
code unmodified:

- The LLM/Coordinator cannot supply a trusted identity. `AuthorizationContext`/
  `VerifiedIdentity` are built from the caller-asserted `tenant_id`/`roles`
  passed into `MultiAgentState` at graph-invoke time, the same "demo
  identity" convention `langgraph_experiment.identity.DemoIdentity` already
  uses (see that module's docstring for why this experiment, like B, is
  about graph mechanics, not re-proving the JWT boundary).
- This experiment never touches the MCP transport at all. `business_
  agent.py`'s reused nodes call `rag.mcp.business.store` in-process,
  exactly as B does (see B's README's "Scope decisions"), so there is no
  internal service token, no caller-JWT-forwarding question, to even ask
  here.
- Authorization is server-side, inside `rag.mcp.business.store._is_
  authorized`/`_lookup_authorized`, unmodified.
- Approval is trusted runtime state: `wait_for_approval` (reused from B)
  re-checks a resume payload's `approver_roles` against
  `config.mcp.business_actions.approval_roles` before ever treating a
  `"decision": "approve"` claim as real. Proven directly by `tests/
  test_case4_business_mutation.py::test_approval_claim_without_a_real_
  approver_role_is_denied`.
- Transition rules are deterministic (`rag.mcp.business.store.
  _VALID_TRANSITIONS`/`_SENSITIVE_TRANSITIONS`), untouched.
- The write action is not retried uncontrollably: it is reached via a
  bare, non-`Send` conditional edge with no retry policy attached, and
  the mutation branch structurally never re-enters the coordinator/critic
  retry loop (that loop only exists on the non-mutation side of
  `route_after_coordinator`).
- Cross-tenant access still fails closed. `test_case2_business_read_
  only.py`'s second test proves a same-tenant-wrong-role denial reports
  `"case_not_found_or_denied"` with **zero** case content leaked into the
  answer (`forbidden_answer_substrings` check, also run for every
  authorization-denial row in `eval/multi_agent_gold.jsonl`).

**Nothing above is reimplemented inside this package's prompts or
nodes**. Every rule is enforced exactly once, in `rag.mcp.business.
store`, regardless of which of A/B/C called into it.

## Bounded execution

| Bound | Constant | Prevents |
|---|---|---|
| Coordinator rounds | `limits.MAX_COORDINATOR_ROUNDS` (2: round 0 + one retry round) | An unbounded coordinator↔critic ping-pong. |
| Knowledge-specialist retries | `limits.MAX_KNOWLEDGE_RETRIES` (1) | Retrying a specialist indefinitely on persistent failure. |
| Business-specialist retries | 0, by design (`evidence_critic` never requests one) | Wasting a tool call on a deterministic lookup that cannot change. |
| Case-read transient-failure retries | `RetryPolicy(max_attempts=3, retry_on=(TransientCaseStoreError,))`, reused from B | An infinite retry on a genuinely flaky (simulated) backend call. |
| Parallel fan-out width | Exactly the specialists named in `selected_specialists` (at most 2) | Uncontrolled spawning. There is no code path that can `Send` to more than the two known specialist entry points. |

Every terminated run has a recorded `termination_reason`
(`synthesized`/`max_rounds`/`insufficient_evidence`/`case_not_found_or_
denied`/`executed`/`already_in_status`/`invalid_transition`/`approval_
required`/`rejected`/`approval_denied_insufficient_role`/`invalid_
request`), never silently `None` on a completed run.

## Failure handling

Explicitly tested, per the spec's required list:

| Scenario | Where it's proven |
|---|---|
| Coordinator picks the wrong specialist | Guarded against by keeping routing deterministic and regex-based rather than LLM-driven (see "Scope decisions"); `tests/test_coordinator_routing.py` pins the exact routing decision for representative phrasings, including a real bug caught by this package's own tests (a too-broad "what is" cue misrouting a pure business question, see ISSUES.md-adjacent commentary below). |
| Specialist structured output malformed | N/A by construction. No specialist parses LLM-generated JSON; both specialists' outputs are typed Python return values from deterministic/tool code, not parsed model output. |
| Specialist times out | `knowledge_agent`'s own `try/except`, `tests/test_case5_specialist_failures.py`. |
| Specialist tool fails | Same as above; business-side transient failures use the reused `RetryPolicy`. |
| Authorization denied | `tests/test_case2_business_read_only.py`'s second test; `eval/multi_agent_gold.jsonl`'s `bq-2-auth-denied`/`bq-3-cross-tenant-denied` rows. |
| MCP unavailable | N/A. This experiment never calls MCP transport at all (in-process business-store calls only, see "Business-action safety"). |
| No evidence returned | `final_synthesis`'s explicit zero-evidence guard; `fail-1-knowledge-timeout` gold row. |
| Conflicting knowledge/business evidence | Deliberately out of scope. See "Evidence Critic" above. |
| Coordinator reaches max rounds | `tests/test_case5_specialist_failures.py::test_knowledge_agent_retries_once_then_gives_up_within_the_bound`. |

## Checkpointing

Reuses the identical durable-SQLite-checkpointer story B already proved
(`checkpointer.py`, pointed at this package's own `data/checkpoints.
sqlite` so the two experiments' demo threads never collide). Demonstrated
at minimum:

- **State persistence**: `cli/run_demo.py state <thread_id>` prints a
  thread's full checkpointed state and any pending interrupt.
- **Inspecting intermediate state**: the same command, run against a
  thread paused mid-mutation (`termination_reason` absent,
  `pending_action.approval_state == "pending"`).
- **Resuming a partially completed workflow**: `cli/approval_cli.py
  approve <thread_id> ...` in a *separate process invocation* from the
  one that started the run. The same real cross-process durability B's
  `test_restart_resume.py` proves, inherited here for free since the
  write branch's nodes (and their `interrupt()` call) are the identical,
  unmodified functions.

Checkpointing is **not** forced onto the simple knowledge-only/business-
read-only routes any more than it is in B. Every route uses the same
one checkpointer; there's no special-cased "skip checkpointing for simple
requests" logic, matching the spec's "don't force checkpointing into
every simple request" guidance by simply never needing to special-case
it in the first place (checkpoint-write overhead is small and uniform;
see the `rag`/`mixed` timing commands for real numbers once run against
live infra).

## Evaluation results

`python -m multi_agent_experiment.eval.run_multi_agent_eval`: a 12-row
gold set (`eval/multi_agent_gold.jsonl`), zero external services (a
small synthetic knowledge base plus the real, self-contained `rag.mcp.
business.store`), deterministic, no LLM judge. Real output, not
fabricated:

```
num_examples: 11
routing_accuracy: 1.000
unnecessary_agent_rate: 0.000
termination_reason_accuracy: 1.000
answer_correctness: 1.000
citation_grounding: 1.000
mean_tool_call_count: 0.909
mean_llm_call_count: 0.455
mean_latency_ms: ~10
security_failures: 0  (must be 0)
termination_reasons: {'synthesized': 5, 'case_not_found_or_denied': 2,
                       'executed': 1, 'invalid_request': 1, 'rejected': 1,
                       'max_rounds': 1}
```

(`num_examples: 11` because one row's approval-flow scenario shares a
thread lifecycle with another. See the gold file's `mut-1-executed`/
`mut-3-rejected` rows for the two outcomes of the same mutation query
under different resume decisions.) Covers, per the spec's required
category list: knowledge-only (×3, including an explicit "identity
present but business must not run" unnecessary-agent check),
business-only (×1), mixed (×1), business mutation (×3: executed,
rejected, invalid-request), authorization denial (×2: same-tenant
wrong-role, cross-tenant), specialist failure (×1). `security_failures`
is a zero-tolerance ceiling, not a quality score, matching this
repo's own `duplicate_sensitive_field_miss_rate`/`sensitive_data_false_
redaction_rate` convention (CLAUDE.md's field-level-safety milestone) of
treating a security metric as a regression guard, not a number to
optimize incrementally.

This eval set was **not** tuned to make multi-agent look better. It was
written to hit the required category list, and every row's expected
routing/answer/termination was derived from reading the actual
`rag.mcp.business.store` seed data and this graph's own deterministic
classification rules, not adjusted after the fact to pass (the one
adjustment made during development, `mx-1`'s expected citation count
2→3, was a correction to a wrong hand-written expectation: the tiny
synthetic KB's "support" keyword legitimately matches two documents,
not a loosening of a real check; see the row's own `notes` field).

## Comparing three architectures

`cli/compare_harness.py write-action` prints the full table below;
`cli/compare_harness.py rag "<query>"` / `mixed "<query>"` additionally
time real queries (needs `make up` + native Ollama + an ingested
dataset. Not run against live infra here, see "What wasn't
verified" below).

| Axis | A: custom harness | B: LangGraph single-agent | C: LangGraph multi-agent |
|---|---|---|---|
| Code complexity | ~150 lines for the write-action path incl. one Python while-loop; directly steppable in a debugger. | ~110 lines across nodes.py/graph.py for the same path; control flow lives in the Pregel runtime, not a visible loop. | ~40 lines of new orchestration code (coordinator/merge/critic) plus the business branch reused unmodified from B, more topology, but almost no new business logic. |
| State visibility | `AgentState` is one pydantic object; no built-in history within a run. | `GraphState` checkpointed after every node. Full history via `get_state_history()`. | Same as B, plus per-specialist fields (`knowledge_status`, `business_status`) make it visible *which* agent contributed what. |
| Branching | Plain if/elif; not queryable at runtime. | `add_conditional_edges()`. Introspectable via `get_graph()`. | Same, plus `Send`-based dynamic fan-out: the actual *set* of specialists dispatched is a runtime decision. |
| Retry handling | None built in. | `RetryPolicy` on a node. Declarative retry with backoff. | `RetryPolicy` (reused) plus a second, higher-level retry: `evidence_critic` can bounce a run back through the coordinator to retry one specialist with widened parameters, a retry loop a single node's `RetryPolicy` cannot express. |
| Checkpointing | None. | Every node transition durably checkpointed. | Same, with one added gotcha this package hit for real: a downstream join node fed by parallel branches of *different* length must be declared `add_node(..., defer=True)` or it fires once per branch instead of once overall. See ISSUES.md. |
| Pause/resume | Not supported. | `interrupt()`/`Command(resume=...)`: the core primitive B exists to demonstrate. | Same mechanism, reused unmodified from B's write branch. |
| Human approval | Must be supplied before the run starts. | A run can pause mid-flight and durably wait. | Same as B (identical code). |
| Latency | One process, one call stack. | Small per-step checkpoint-write overhead. | Same per-step overhead as B, offset by genuine parallel dispatch on a mixed query. |
| Debugging | Standard Python debugger end to end. | `get_state(config).tasks[i].error` surfaces a failed node's exception without a live debugger. | Same, plus `tool_call_log`/`critic_notes` narrate *why* a specific specialist's contribution was or wasn't used. |
| Security surface | JWT boundary + `AuthorizationContext` + field redaction, all in production code. | Same production code, reused; demo identity is caller-asserted (see B's README). | Identical to B. This experiment adds a fourth layer, structural tool isolation (see above), on top, not a replacement. |
| Testability | Tests call `run_agent()` directly with fakes; fully synchronous. | Tests inject fakes via `GraphDeps`, assert on checkpoint state directly. | Same `GraphDeps` injection, plus a static import-boundary test that a single-agent graph has no equivalent need for. |
| Maintainability | One file, one loop. Easiest to read end to end for its current scope. | Reading a run means reading `graph.py`'s edges *and* every node function. | Same as B, plus understanding "why did X happen" now sometimes means reading two specialists' interaction, not one node's logic, genuinely more moving parts to hold in your head. |
| Coordination | N/A. One agent, one tool-selection loop; "coordination" is just sequential tool calls within one bounded loop. | N/A. Single route per run, no specialist-to-specialist handoff. | A real coordinator/router decision, parallel fan-out for independent specialists, and a merge step reconciling two genuinely different evidence shapes into one grounded answer. |

**When is each architecture justified?**

- **A (custom harness)** is justified for everything production actually
  needs today: synchronous request/response, pre-authorized write
  actions, bounded decomposition/tool-use within one agent. It is the
  simplest code, the easiest to debug live, and needs zero extra
  infrastructure. Nothing in this experiment changes that assessment;
  if anything it reinforces it, since C's added complexity buys real
  value only on the specific "mixed, needs two independent, differently-
  toolable specialists" shape, which production doesn't structurally
  have today (see "What should NOT be merged" below).
- **B (single-agent LangGraph)** is justified specifically where A has a
  real structural gap: pausing an in-flight run to wait on an external
  event (a human decision) and surviving a process restart while paused.
- **C (multi-agent LangGraph)** is justified only when B's justification
  *and* a genuine need for two-or-more agents with different tool access
  running against the same query both hold, e.g. a real deployment
  where "knowledge" and "business-case" evidence must be fetched by
  different services/teams/security boundaries, or where the two would
  otherwise need to be dispatched over a network in parallel (not the
  case here. Both specialists are in-process function calls, so the
  parallelism win is real but modest; see "What I learned" below for
  where a *networked* multi-agent system would look different).

**Not concluded**: that LangGraph or multi-agent is automatically
superior. C is measurably more code, more moving parts, and, for every
scenario CASE 1/2/4 in this experiment's own required list, strictly
more overhead than A or B for no benefit, since those routes only ever
needed one specialist. C's only real win is CASE 3 (mixed), where two
independent tool calls can run concurrently instead of sequentially.

## What wasn't verified

Honestly scoped, matching this repo's own convention of flagging gaps
rather than implying more was checked than was:

- `cli/compare_harness.py rag`/`mixed` were written and import-checked
  but **not run against a live Postgres + Ollama stack** here.
  Docker Desktop was not running when this experiment was built (only
  native Ollama was reachable). The qualitative comparison table above
  is reasoned from B's own already-measured overhead plus this graph's
  added hop count, not fabricated, but there is no real measured
  `Latency` number for A vs. B vs. C on this machine yet.
- The `mixed` timing subcommand's "sequential estimate" is, by its own
  docstring, an estimate built from two separate real measurements, not
  a second real measurement of an actual sequential graph. No throwaway
  sequential-variant graph was built solely to produce one number.

**Next recommended step**: bring up `make up` + an ingested `techfusion`
dataset and run `compare_harness.py rag "<question>" --dataset-id
techfusion` and `compare_harness.py mixed "<mixed question>" --dataset-id
techfusion --tenant tenant_alpha --role tenant_alpha_operator
--business-only-query "What is the status of CASE-1001?"` for real
numbers, the same "flagged, not fabricated" pattern B's own README used
for its unrun `rag` comparison when it was first written.

## Security analysis

- **Authorization**: enforced exactly once per concern, in exactly one
  place, regardless of caller. `RetrievalPipeline`/`PgVectorStore` for
  document-level ACL/freshness/trust (knowledge branch), `rag.mcp.
  business.store` for case-level ACL/transition/approval (business
  branch). This graph adds zero authorization logic of its own.
- **Field-level redaction**: inherited for free. `knowledge_agent`'s
  evidence is already the sanitized output of `RetrievalPipeline.
  retrieve()` (field redaction + injection flagging already applied
  internally) by the time this graph ever touches it.
- **No secret/credential ever enters shared state**: see "Shared graph
  state" above.
- **Approval cannot be forged**: `wait_for_approval`'s reused role
  re-check (see "Business-action safety").
- **Tool isolation is structural, not advisory**: see "Tool isolation"
  above. This is the one genuinely new security-adjacent property this
  experiment adds beyond what A/B already had, and it is a **blast-
  radius** control (limiting what a *buggy* specialist could reach), not
  a replacement for the authorization boundary that holds regardless.
- **Residual, accepted risk, same as B**: identity is caller-asserted
  (`tenant_id`/`roles` passed directly into `MultiAgentState`), not a
  verified JWT. This experiment is about graph mechanics, not
  re-proving the JWT boundary `rag.api.auth`/production's `security.
  auth.enabled=True` mode already covers (see B's README's identical
  scoping decision).
- **No new attack surface for the write path**: the mutation route
  reuses B's exact, already-security-reviewed code (`langgraph_
  experiment.nodes.validate_write_request`/`wait_for_approval`/
  `make_execute_write_action_node`) unmodified; this graph only decides
  *whether* to route there, never *what happens* once it does.

## Interview learning output

### Twenty concepts

1. **What makes a system multi-agent.** More than one agent with
   genuinely different responsibilities and tool access, coordinated
   through explicit handoffs/merges, not just "the same capabilities,
   different prompts" (the spec's own stated anti-pattern). Here:
   Knowledge Agent and Business Agent literally cannot call each other's
   tools (verified statically), which is the concrete, checkable version
   of "genuinely different."
2. **Multi-agent vs. one agent with many tools.** Production's own A
   already gives one agent four-plus tools (`rag.agent.tool_schemas`)
   and picks among them via one LLM-driven selection step. That's *not*
   multi-agent: one reasoning loop, one shared tool budget, one
   decision-maker. Multi-agent adds a second (or more) independent
   locus of control: here, the Business Agent's read/write path has its
   own internal routing (`select_case_tool`, `route_after_validate`)
   that the Coordinator never sees or overrides.
3. **Why LangGraph is useful for multi-agent orchestration.** Nodes as
   data (inspectable topology via `get_graph()`), `Send` for dynamic
   fan-out to a *subset* of known specialists chosen at runtime,
   checkpointing that survives a process restart mid-coordination, and
   `interrupt()` for a specialist (here, the Business Agent) to pause
   the *whole* multi-agent run on an external event, none of which a
   plain Python function-call chain gives you for free.
4. **Coordinator/router pattern.** One node whose only job is "decide
   who handles this, and how many of them," never doing the work itself.
   `orchestrator.coordinator` never fetches evidence or touches a case.
   Only classifies and re-parameterizes on retry.
5. **Specialist-agent pattern.** A node (or subgraph) with a narrow,
   named responsibility and its own tool budget, testable and reasoned
   about in isolation. `knowledge_agent.py`/`business_agent.py` each
   import zero cross-specialist code.
6. **Conditional routing.** `add_conditional_edges` mapping a decision
   function's return value to one or more destinations. Here, `route_
   after_coordinator` returns either a bare string (mutation) or a list
   of `Send` objects (1 or 2 specialists), one function handling all
   four topologies.
7. **Shared graph state.** `MultiAgentState`: every specialist's
   contribution lands in a namespaced slice of one shared, checkpointed
   dict; see field-ownership discipline above.
8. **Agent-private state.** A specialist's internal reasoning that never
   needs to leave its own node function (e.g. exactly how `knowledge_
   agent` decides to catch vs. propagate an exception), not exposed as
   a state field unless a sibling genuinely reads it.
9. **Tool isolation.** See the dedicated section above: the clearest,
   most concretely testable concept this experiment produced.
10. **Handoff.** The Coordinator's `Send` dispatch *is* the handoff:
    an explicit, typed transfer of control (and the relevant slice of
    state) to a named specialist, never an implicit fallthrough.
11. **Parallel branches.** `Send`-based fan-out to two specialists in one
    superstep. See "Parallel execution" above, including the real bug
    (unequal branch length breaking a naive join) found building this.
12. **State merge.** `orchestrator.merge`, `defer=True`-gated so it only
    runs once every dispatched branch has actually finished, regardless
    of how many hops each branch took.
13. **Bounded loops.** Two independent counters (`coordinator_round`,
    `retry_count`), each with its own ceiling, so a genuinely
    unresolvable question terminates with a diagnosable reason rather
    than looping or crashing.
14. **Checkpointing.** Reused, unmodified mechanism from B; demonstrated
    for the write path (the one place it's load-bearing, not merely
    nice-to-have) via the identical cross-process restart story.
15. **Failure propagation.** A specialist's own exception never
    propagates past its own node. It's caught and turned into a typed
    status (`knowledge_status="failed"`) the Coordinator/Critic can
    reason about, matching this whole codebase's "never let a tool
    failure crash the request" convention (`rag.agent.tools.
    ToolExecutionError`'s own precedent).
16. **Multi-agent security boundaries.** Structural tool isolation
    (new to this experiment) layered on top of unchanged authorization/
    redaction enforcement (inherited): two different kinds of boundary,
    doing two different jobs; see "Security analysis."
17. **Why multi-agent costs more.** More nodes to traverse even for a
    solo-specialist query (`coordinator → merge → evidence_critic →
    final_synthesis` vs. B's `classify → retrieve → synthesize_rag`),
    more code to read to understand one run, and a real correctness
    pitfall (the fan-in bug) that a single-agent graph structurally
    cannot have, because it never has two branches to join at all.
18. **When multi-agent improves quality.** When two evidence sources are
    genuinely independent and both needed (CASE 3 here), parallel
    dispatch cuts wall-clock time versus sequential tool calls, and a
    named, isolated specialist per evidence source makes "which source
    grounded this claim" traceable in a way one undifferentiated agent's
    tool-call log doesn't as cleanly.
19. **When multi-agent is unnecessary.** Every solo-specialist route in
    this very experiment (CASE 1, 2, 4): multi-agent topology adds
    hops and code for zero benefit when only one specialist was ever
    going to run. The evaluation's own `unnecessary_agent_rate: 0.000`
    is the concrete instrument for catching a regression on this exact
    point.
20. **LangGraph multi-agent vs. custom orchestration.** A hand-rolled
    multi-agent system (plain Python: a router function calling two
    other functions, `concurrent.futures` for parallelism, a dict merge)
    is entirely possible and would be *simpler* for exactly this
    experiment's scope. No checkpointing, no `Send`, no `defer=True`
    gotcha to learn. LangGraph earns its complexity when you also need
    the same things B's README already identified for the single-agent
    case (durable pause/resume, inspectable state history across a
    process boundary) *combined with* multiple specialists. This
    experiment demonstrates both are compatible (the write branch's
    interrupt still works fine reached through a coordinator that also
    routes elsewhere), not that multi-agent alone requires LangGraph.

### Common multi-agent failure modes (and where this experiment did/didn't hit each)

- **Routing mistakes.** Hit for real: an early `_KNOWLEDGE_CUE_RE`
  version matched generic phrases like "what is", misrouting "What is
  the status of CASE-1001?" (a pure business question) into needing the
  Knowledge Agent too. Caught by `tests/test_coordinator_routing.py`
  before it ever reached a scenario test. Fixed by requiring a cue that
  signals "references the knowledge base," not merely "is a question."
- **Duplicated work.** Guarded against directly: a coordinator retry
  round explicitly sets `needs_business=False` so a deterministic
  business read is never redundantly re-run just because the knowledge
  side needed another attempt.
- **Context explosion.** Not a real risk in this design. Each
  specialist's context is exactly its own tool's output; `final_
  synthesis` is the only node that ever sees both, and even then only
  the already-bounded evidence lists, not each specialist's internal
  reasoning.
- **Excessive model calls.** Bounded by construction. At most one LLM
  call (`final_synthesis`) per terminal non-mutation route, regardless
  of how many specialists ran; `mean_llm_call_count: 0.455` in the eval
  report reflects that most routes (business-only, mutation) make zero
  LLM calls at all.
- **Conflicting agents.** Structurally avoided (disjoint state
  ownership, see "Parallel execution"), but semantic conflict between
  what the two specialists *report* is explicitly out of scope (see
  "Evidence Critic" above), a real, acknowledged gap, not a solved
  problem.
- **Recursive loops.** Bounded by two independent counters (see "Bounded
  execution").
- **Hidden latency.** The fan-in bug (see ISSUES.md) manifested exactly
  as hidden latency/cost before it was found: a doubled LLM call that a
  naive timing check might not have caught if the test suite hadn't
  asserted on call *count*, not just wall-clock time or final answer
  content.
- **Difficult debugging.** Real, not hypothetical. Diagnosing the
  fan-in bug required `stream_mode="debug"` to see step numbers, since
  `stream_mode="updates"` (the more obvious choice) emits one chunk per
  node completion and does not itself reveal which nodes shared a
  superstep (see ISSUES.md's full diagnosis and `tests/test_case3_
  mixed_parallel.py`'s own docstring for why `"updates"` was tried and
  rejected first).
- **Shared-state corruption.** Avoided by the field-ownership discipline
  in `state.py`; the one near-miss was the fan-in bug's *symptom* being
  state-shaped (an early `merge` overwriting itself), not a genuine
  write-write conflict LangGraph would have raised an error for.
- **Security expansion.** Explicitly checked against: tool isolation is
  additive (a narrower blast radius), not a widening of what any code
  path can reach. See "Security analysis."

### Twenty likely interview questions

1. **Q: What's the concrete difference between "multi-agent" and "one
   agent with more tools"?**
   A: Independent loci of control with separated tool access, not just a
   bigger tool list on one decision-maker. Here, Knowledge and Business
   are separately testable, separately failable, and, provably, via a
   static import check, structurally unable to call each other's tools.
2. **Q: Why does `route_after_coordinator` return `Send` objects instead
   of plain node-name strings?**
   A: `Send` lets a conditional edge dispatch to a *dynamically chosen
   subset* of nodes as genuinely parallel tasks in one superstep; a plain
   string (or list of strings in a `path_map`) picks exactly one static
   destination.
3. **Q: What broke the first version of the parallel "mixed" route?**
   A: The join node (`merge`) had unequal-length incoming branches
   (Knowledge: 1 hop; Business: 3 hops) and no `defer=True`, so LangGraph
   ran it once per branch instead of once overall, doubling an LLM call.
4. **Q: How does `defer=True` actually fix that?**
   A: It tells LangGraph to hold that node's execution until every
   currently-pending task in the run has finished, rather than firing as
   soon as any one predecessor edge delivers a write.
5. **Q: How would you have found that bug without a debugger?**
   A: `graph.stream(..., stream_mode="debug")`. It surfaces a numeric
   `step` per task event, so two nodes sharing a step number is direct
   proof of same-superstep parallelism (or the lack of it).
6. **Q: Why not use `stream_mode="updates"` for that same check?**
   A: It emits one chunk per node *completion*, not per superstep, so
   two genuinely parallel nodes still show up as two separate chunks;
   confirmed empirically before writing any test around it.
7. **Q: How do you prevent two parallel branches from corrupting shared
   state?**
   A: Field-ownership discipline: each specialist only ever writes its
   own namespaced keys, plus `Annotated[list, operator.add]` reducers
   on the few fields that genuinely can be written more than once
   (across a retry), so LangGraph accumulates rather than conflicts.
8. **Q: Why is the Business Agent never retried, even though the
   Knowledge Agent is?**
   A: Its read is a deterministic lookup against `rag.mcp.business.
   store`. An identical second call returns an identical result, so a
   retry would only cost a tool call, never change the outcome.
9. **Q: What stops the Coordinator from retrying forever?**
   A: Two independent bounds: `MAX_COORDINATOR_ROUNDS` (how many times
   the coordinator itself may run) and `MAX_KNOWLEDGE_RETRIES` (how many
   retries the critic may request), checked together before a retry is
   ever granted.
10. **Q: Why does a retry round set `needs_business=False`?**
    A: The business branch already produced its final result on round 0;
    re-dispatching it on a knowledge-only retry would be wasted work, not
    a correctness fix. The retry loop only ever widens what actually
    needs another attempt.
11. **Q: How is tool isolation actually verified, not just claimed?**
    A: `tests/test_tool_isolation.py` parses each specialist module's
    source with `ast` and asserts the forbidden module names never
    appear among its real `import`/`from` statements, a structural
    check, not a runtime permission flag.
12. **Q: Does tool isolation replace backend authorization?**
    A: No. `update_case_status`'s tenant/role/transition/approval
    checks would still run and still deny an unauthorized request even
    if a specialist somehow imported the wrong module. Isolation limits
    blast radius; it isn't the security boundary itself.
13. **Q: Why is the Coordinator's classification deterministic (regex)
    instead of LLM-driven?**
    A: Keeping routing reproducible is what makes the retry/parallel/
    interrupt mechanics this experiment exists to teach testable at all;
    production's own agent (`rag/agent/decisions.py`) already
    demonstrates LLM-driven structured routing, so repeating that here
    would teach nothing new about multi-agent topology specifically.
14. **Q: What's a concrete case where that deterministic router got it
    wrong, and how was it caught?**
    A: An early cue-word list matched generic phrases like "what is,"
    misrouting a pure business question ("What is the status of
    CASE-1001?") into also needing the Knowledge Agent. Caught by a
    unit test asserting the exact expected routing flags, before it ever
    reached a scenario-level test.
15. **Q: How does the write-action branch avoid the mutation being
    retried uncontrollably?**
    A: It's reached via a bare (non-`Send`) conditional edge with no
    retry policy, and it structurally never re-enters the coordinator/
    critic loop. That loop only exists on the non-mutation side of the
    router, so there is no path back into a mutation attempt at all.
16. **Q: Where does approval actually get checked?**
    A: Inside `wait_for_approval` (reused unmodified from the
    single-agent experiment), which re-checks the resume payload's
    `approver_roles` against configured approval roles: a `"decision":
    "approve"` claim alone is never trusted.
17. **Q: What would need to change to make this a real, networked
    multi-agent system instead of in-process function calls?**
    A: Each specialist would become a separate service call (likely
    behind the existing MCP transport this experiment deliberately
    bypasses, see "Scope decisions"), which would change the
    parallelism math meaningfully: today's win is concurrent I/O-bound
    calls in one process; a networked version adds real network latency
    per hop that parallel dispatch would hide more of.
18. **Q: Why wasn't a semantic evidence-conflict check built into the
    Evidence Critic?**
    A: It would need either another LLM call (reintroducing exactly the
    flakiness the deterministic design elsewhere avoids) or a keyword
    heuristic prone to false positives/negatives on the nuanced kind of
    claim a conflict check exists to catch, scoped out explicitly, not
    silently skipped.
19. **Q: How would you extend this to a third specialist?**
    A: Add its own isolated module (own tools, own tests, own static
    isolation check), add it to `route_after_coordinator`'s `Send` list
    under its own `needs_*` flag, add an edge into `merge`, and update
    `evidence_critic`/`final_synthesis` to read its status/evidence
    field; no existing specialist's code changes.
20. **Q: When would you NOT reach for this multi-agent shape?**
    A: When one agent with several tools and a single decision loop
    already covers the scenario, which is true for three of this
    experiment's own five required scenarios (knowledge-only,
    business-only, mutation), where the multi-agent topology adds hops
    and code for zero behavioral benefit over a single-agent design.

## What I learned that would be difficult to learn from the custom agent alone

- **The `defer=True` fan-in gotcha itself.** A's single-loop architecture
  has no concept of "parallel branches of different length converging on
  one node". This failure mode simply cannot occur in a system with no
  parallel dispatch primitive. Finding and fixing it required actually
  building a multi-agent graph, not reasoning about LangGraph in the
  abstract.
- **`stream_mode="debug"` vs. `"updates"`** for actually verifying
  parallel execution happened, versus merely *hoping* it did because the
  code "looks" parallel, a distinction that only matters once there's
  more than one thing that could be running concurrently.
- **Static tool-isolation as a first-class, testable property.** A's
  single agent has one tool budget, so "can this agent reach that tool"
  is trivially "yes". The question, and the AST-based test that answers
  it, only becomes meaningful once there's more than one agent with
  *different* tool access to keep separate.
- **The real cost/benefit of parallel dispatch for two in-process tool
  calls.** Both specialists here are synchronous Python function calls,
  not network requests. The parallelism win is real (concurrent
  I/O-bound retrieval + business-store lookup) but modest compared to
  what a genuinely networked multi-agent system (each specialist as a
  separate service call) would show; this experiment makes that
  distinction concrete rather than assumed.
- **Coordinator retry is a genuinely different primitive from a single
  node's `RetryPolicy`.** `RetryPolicy` retries *one node* with the
  *same* input; `evidence_critic`'s retry changes a *parameter*
  (`top_k`) and re-dispatches through the Coordinator, a strictly
  richer retry shape a single-agent design has no reason to need.

## What should NOT be merged into `main` unless there's a real need

Matching this repo's own precedent for `langgraph_experiment/`/`k8s/`
(both explicitly kept off `main`, learning exercises, not production
candidates):

- **This entire package.** Production has no current requirement for two
  independently-toolable specialists to run in parallel against one
  query. `rag.agent.graph.run_agent`'s single bounded loop already
  covers every scenario this repo's real gold sets (`techfusion_gold.
  jsonl`, `agentic_extension_gold.jsonl`, `specialized_tool_gold.
  jsonl`) exercise, without the added topology/checkpoint-overhead cost
  this experiment's own comparison table documents honestly.
- **`defer=True` as a "just add it everywhere" reflex.** It was the
  correct fix for *this* unequal-length join; reaching for it on every
  join node without checking whether branches are actually unequal
  length would be premature-optimization-shaped superstition, not an
  informed choice.
- **The `Send`-based routing pattern**, specifically, unless a real
  production route needs to dispatch to a *dynamically-sized* set of
  parallel specialists. Today's agentic RAG milestone's own tool-
  selection loop (`rag/agent/decisions.py`) already covers "which of N
  known tools to call," sequentially, which is sufficient for every
  real scenario documented in CLAUDE.md's agentic-rag section.
- **The synthetic knowledge base fakes** (`cli/scenario_walkthrough.py`,
  `eval/run_multi_agent_eval.py`), deliberately toy, deliberately
  disconnected from the real `techfusion` corpus; useful for this
  experiment's zero-setup reproducibility goal, never a stand-in for a
  real evaluation against real data.

## Setup and run commands

Uses the **same** `.venv-langgraph/` virtual environment
`langgraph_experiment/` already built (see that package's README's
"Dependency isolation" section for the full `langchain-core` version
conflict this venv exists to work around); no new venv, no new
dependency to install:

```bash
source .venv-langgraph/Scripts/activate      # Windows/Git-Bash
# or: source .venv-langgraph/bin/activate    # macOS/Linux
```

```bash
# All 5 required scenarios, zero external services, one command:
python -m multi_agent_experiment.cli.scenario_walkthrough

# Evaluation harness, zero external services:
python -m multi_agent_experiment.eval.run_multi_agent_eval
python -m multi_agent_experiment.eval.run_multi_agent_eval --out report.json

# Tests, zero external services:
python -m pytest multi_agent_experiment/tests -v

# Against real infra (make up + native Ollama + an ingested dataset):
python -m multi_agent_experiment.cli.run_demo start "What is the password rotation policy?" \
    --dataset-id techfusion
python -m multi_agent_experiment.cli.run_demo start "Please close CASE-2001" \
    --tenant tenant_beta --role tenant_beta_operator --no-retrieval
python -m multi_agent_experiment.cli.approval_cli list
python -m multi_agent_experiment.cli.approval_cli approve <thread_id> \
    --approver-subject alice --approver-roles case_status_approver

# Architecture comparisons:
python -m multi_agent_experiment.cli.compare_harness write-action   # no infra needed
python -m multi_agent_experiment.cli.compare_harness rag "<query>" --dataset-id techfusion
python -m multi_agent_experiment.cli.compare_harness mixed "<mixed query>" \
    --dataset-id techfusion --tenant tenant_alpha --role tenant_alpha_operator \
    --business-only-query "What is the status of CASE-1001?"
```

## Testing

28 tests (`multi_agent_experiment/tests`), all passing, **zero external
services**, same "no Postgres/Ollama needed" discipline B's own test
suite established, since the business branch is self-contained and the
knowledge branch's tests use a narrow `FakePipeline`/`FakeLLM`.

```bash
python -m pytest multi_agent_experiment/tests -v
```

| File | Proves |
|---|---|
| `test_probe_langgraph_primitives.py` | The five LangGraph primitives this package's design depends on, against a minimal graph independent of this package's own topology, including the fan-in/`defer=True` regression pair. |
| `test_tool_isolation.py` | AST-based static proof that each specialist never imports the other's tool surface. |
| `test_coordinator_routing.py` | Pure, deterministic routing/extraction, no graph invocation. |
| `test_case1_knowledge_only.py` … `test_case5_specialist_failures.py` | The five required scenarios, end to end through the compiled graph. |

## Deviations from a literal reading of the prompt spec

- **Section 3's optional Evidence Critic was added**, scoped narrowly
  (see "Evidence Critic: why it earns its place").
- **Section 4's topology diagrams are followed exactly**: knowledge-only
  and business-only routes are solo `Send` dispatches (not literally
  "sequential from Coordinator" as plain edges, but the observable
  behavior (one specialist runs, one path to Merge/Synthesis) is
  identical); mixed is a genuine two-way parallel `Send`; business
  mutation bypasses Merge/Synthesis entirely, going straight from the
  Coordinator to the existing MCP action path, matching the spec's own
  diagram for that case exactly.
- **No decomposition into subquestions** (`subquestions` mentioned in
  section 5's potential shared-state list): every route dispatches each
  selected specialist with the *original* query, never a decomposed
  subquestion. Adding decomposition would mean either a third,
  non-deterministic LLM call (reintroducing the JSON-parsing flakiness
  B's own design explicitly avoids) or a second deterministic heuristic
  layered on top of an already-deterministic router, judged not to add
  teaching value proportional to the complexity, matching this
  experiment's own "don't build infrastructure without a demonstrated
  need" convention (the same one CLAUDE.md documents for Redis/a second
  ingestion worker in the main system).
