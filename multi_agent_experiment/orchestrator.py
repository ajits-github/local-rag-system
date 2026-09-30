"""Coordinator, merge, critic, and final-synthesis nodes: the "sees both specialists" tier.

Unlike `knowledge_agent.py`/`business_agent.py`, this module is allowed to
read both `knowledge_*` and `case_*`/`business_*` state -- it is the
orchestration layer, not a specialist, and never calls a tool directly
(no `RetrievalPipeline`, no `rag.mcp.business.store` import anywhere in
this file). The only "tool-shaped" import here is
`_looks_like_case_mutation_request` (a pure regex classifier reused from
`rag.agent.graph`, not a call into any backend) and
`_extract_case_id`/`_extract_target_status` (reused regex helpers from
`langgraph_experiment.nodes`) -- text classification, not a business
mutation. See `tests/test_tool_isolation.py` for the static check that
even this "sees everything" module never imports `rag.mcp.business.store`
directly (it can read case data already placed in state by
`business_agent.py`'s reused nodes, but cannot itself fetch or mutate a
case).
"""

from __future__ import annotations

import re

from langgraph_experiment.nodes import _extract_case_id, _extract_target_status
from langgraph_experiment.state import Citation
from langgraph_experiment.wiring import GraphDeps
from multi_agent_experiment.limits import (
    DEFAULT_KNOWLEDGE_TOP_K,
    MAX_COORDINATOR_ROUNDS,
    MAX_KNOWLEDGE_RETRIES,
    RETRY_KNOWLEDGE_TOP_K,
)
from multi_agent_experiment.state import MultiAgentState, ToolCallRecord
from rag.agent.graph import _looks_like_case_mutation_request

#: A deliberately small, literal cue list -- not an LLM classification, for
#: the same reason `langgraph_experiment.nodes`' own routing/extraction is
#: deterministic (see that module's docstring): keeping the coordinator's
#: routing decision reproducible is what makes the bounded-retry/parallel-
#: branch/interrupt mechanics this package exists to teach testable at
#: all, rather than confounded by LLM JSON-parsing flakiness. Production's
#: own agent (`rag/agent/decisions.py`) already demonstrates LLM-driven
#: structured routing; duplicating that here would teach nothing new about
#: multi-agent *topology*, which is this experiment's actual subject --
#: see README.md's "Scope decisions" for the full reasoning.
#: Generic interrogative openers ("what is", "how does", "explain") were
#: deliberately tried and dropped here: they fire on almost any question,
#: including a purely business one ("What is the status of CASE-1001?"),
#: making `needs_knowledge` true far too often -- confirmed directly by a
#: failing test during this experiment's own development (see
#: `tests/test_coordinator_routing.py`). Only cues that specifically
#: signal "this references the knowledge base/a document," not "this is
#: a question," belong here.
_KNOWLEDGE_CUE_RE = re.compile(
    r"\b(polic(?:y|ies)|documentation|docs?|according to|guide(?:line)?|procedure|"
    r"spec(?:ification)?|timeout|threshold)\b",
    re.IGNORECASE,
)


def coordinator(state: MultiAgentState) -> dict:
    """Coordinator/router node: decide which specialist(s) this query needs.

    Round 0 (the normal case, `coordinator_round` unset/`0`): full,
    deterministic classification --

    1. `is_case_mutation` via the exact same reused
       `_looks_like_case_mutation_request` production's own agent uses
       for this decision (`rag.agent.graph`), so this experiment's
       mutation-detection never drifts from production's.
    2. `case_id`/`requested_new_status` extraction, reused from
       `langgraph_experiment.nodes` (the same regexes the single-agent
       experiment's business branch already relies on).
    3. `needs_business` -- `True` whenever a case id is present or the
       query is a mutation request.
    4. `needs_knowledge` -- `True` whenever a knowledge cue word/phrase
       is present, *or* there is no case id at all (a bare question with
       no case reference defaults to a knowledge question rather than
       being routed nowhere).

    This is a genuine router decision, not a coin flip: it is what
    produces the four distinct topologies in README.md's architecture
    diagram (knowledge-only, business-read-only, mixed parallel, business
    mutation) from one shared entry node.

    A **retry round** (`coordinator_round >= 1`, only ever reached via
    `evidence_critic`'s `"retry"` edge) does **not** re-classify the
    query. It only widens `knowledge_top_k` and forces `needs_business`
    back to `False`, so `routing.route_after_coordinator`'s next `Send`
    fan-out dispatches to `knowledge_agent` alone -- a business read is
    deterministic, so re-running it on retry would waste a tool call and
    change nothing (see `evidence_critic`'s own docstring for the same
    point, stated from the critic's side). `selected_specialists` (used
    by the critic and by the evaluation harness to know what the
    *original* routing decision was) is deliberately left untouched on a
    retry round.
    """
    round_ = state.get("coordinator_round", 0)
    if round_ == 0:
        query = state["original_query"]
        is_mutation = _looks_like_case_mutation_request(query)
        case_id = _extract_case_id(query)
        has_knowledge_cue = bool(_KNOWLEDGE_CUE_RE.search(query))
        needs_business = bool(case_id) or is_mutation
        needs_knowledge = has_knowledge_cue or not needs_business
        selected: list[str] = []
        if needs_knowledge:
            selected.append("knowledge")
        if needs_business:
            selected.append("business")
        return {
            "coordinator_round": 1,
            "is_case_mutation": is_mutation,
            "case_id": case_id,
            "requested_new_status": _extract_target_status(query) if is_mutation else None,
            "needs_knowledge": needs_knowledge,
            "needs_business": needs_business,
            "selected_specialists": selected,
            "knowledge_top_k": DEFAULT_KNOWLEDGE_TOP_K,
        }
    return {
        "coordinator_round": round_ + 1,
        "knowledge_top_k": RETRY_KNOWLEDGE_TOP_K,
        "needs_business": False,
    }


def merge(state: MultiAgentState) -> dict:
    """Join node: combine whatever `knowledge_agent`/business-read branch produced.

    Reached after every non-mutation branch (solo or parallel) -- see
    `graph.py`'s topology. Recomputes `citations`/`tool_call_log` from
    scratch every time it runs, including after a knowledge retry (see
    `state.py`'s module docstring for why these two fields are plain
    overwrites, not reducers, unlike `knowledge_tool_calls`/
    `business_tool_calls` themselves).
    """
    citations: list[Citation] = [
        {
            "chunk_id": item["chunk_id"],
            "case_id": None,
            "source": item["source"],
            "score": item["score"],
        }
        for item in (state.get("knowledge_evidence") or [])
    ]
    case_result = state.get("case_result")
    if case_result is not None:
        citations.append(
            {
                "chunk_id": None,
                "case_id": case_result["case_id"],
                "source": f"case:{case_result['case_id']}",
                "score": None,
            }
        )
    tool_calls: list[ToolCallRecord] = list(state.get("knowledge_tool_calls") or []) + list(
        state.get("business_tool_calls") or []
    )
    return {"citations": citations, "tool_call_log": tool_calls}


def evidence_critic(state: MultiAgentState) -> dict:
    """Evidence Critic node: the one bounded retry decision, and a no-fabrication guard.

    Deliberately narrow, per the "do not add a Critic merely to make the
    system look more multi-agent" instruction this experiment was built
    against. Two concrete responsibilities, both mechanical (no semantic
    conflict detection -- see README.md's "Scope decisions" for why that
    was explicitly considered and left out):

    1. **Bounded retry.** If the Knowledge Agent was selected and came
       back `"no_evidence"`/`"failed"`, and the retry budget
       (`MAX_KNOWLEDGE_RETRIES`) and round budget
       (`MAX_COORDINATOR_ROUNDS`) both still allow it, request exactly
       one retry (routed back through `coordinator`, which widens
       `knowledge_top_k`). The Business Agent is never retried here: its
       read is a deterministic lookup against `rag.mcp.business.store`,
       so an identical second call would return an identical result --
       retrying it would only waste a tool call, not change the outcome.
       `critic_notes` records this reasoning explicitly rather than
       silently skipping it, so a reader of a run's trace can see *why*
       no business retry was attempted.
    2. **No-fabrication guard.** Once retries are exhausted (or were
       never applicable), record a diagnostic note for any specialist
       that still has no usable result. `final_synthesis` independently
       refuses to answer from zero evidence, so this guard is
       redundant-by-design defense in depth, not the only thing standing
       between an empty retrieval and a hallucinated answer -- matching
       this codebase's repeated "a guarantee that depends on one layer is
       not a guarantee" pattern (see CLAUDE.md's field-redaction-marker-
       sanitization entry for the production precedent this mirrors).
    """
    selected = state.get("selected_specialists") or []
    notes: list[str] = list(state.get("critic_notes") or [])
    retry_count = state.get("retry_count", 0)
    round_ = state.get("coordinator_round", 1)

    knowledge_deficient = "knowledge" in selected and state.get("knowledge_status") in {
        "no_evidence",
        "failed",
    }
    retry_budget_remains = retry_count < MAX_KNOWLEDGE_RETRIES and round_ < MAX_COORDINATOR_ROUNDS
    if knowledge_deficient and retry_budget_remains:
        notes.append(
            f"knowledge specialist returned '{state.get('knowledge_status')}' on round "
            f"{round_}; requesting one bounded retry with a wider top_k "
            f"({DEFAULT_KNOWLEDGE_TOP_K} -> {RETRY_KNOWLEDGE_TOP_K})."
        )
        return {"critic_notes": notes, "retry_count": retry_count + 1, "critic_wants_retry": True}

    if knowledge_deficient:
        notes.append(
            f"knowledge specialist still '{state.get('knowledge_status')}' after the retry "
            "budget was exhausted; proceeding without fabricating an answer."
        )

    business_deficient = "business" in selected and state.get("business_status") not in {
        None,
        "ok",
    }
    if business_deficient:
        notes.append(
            f"business specialist returned '{state.get('business_status')}'; this is a "
            "deterministic lookup, so retrying it would not change the outcome."
        )

    return {"critic_notes": notes, "critic_wants_retry": False}


def make_final_synthesis_node(deps: GraphDeps):
    """Build the terminal synthesis node for every non-mutation route.

    Reached from `merge` -> `evidence_critic` -> (`"proceed"`). Builds one
    context block per knowledge chunk plus (when present) one block
    summarizing the authorized, already-sanitized business case result,
    and renders them through the same `config.generation.prompt` template
    production's classic RAG path uses (`deps.rag_prompt_template.
    render(context=..., query=...)`) -- a single real grounded answer over
    both evidence sources for the mixed-query case, matching README.md's
    CASE 3 requirement.

    Refuses to fabricate: if neither specialist produced anything usable,
    returns the fixed "I don't have enough..." answer with
    `termination_reason` set to `"max_rounds"` (retry budget was spent and
    still came up empty) or `"insufficient_evidence"` (no retry was ever
    applicable, e.g. a business-only route that found nothing) --
    distinguishing the two so the evaluation harness/a trace reader can
    tell "we tried harder and still had nothing" from "there was nothing
    to try harder at." Mirrors `langgraph_experiment.nodes.
    make_synthesize_rag_node`'s same no-evidence guard and `deps.llm is
    None` deterministic fallback.
    """

    def final_synthesis(state: MultiAgentState) -> dict:
        knowledge_evidence = state.get("knowledge_evidence") or []
        case_result = state.get("case_result")
        has_case = case_result is not None

        if not knowledge_evidence and not has_case:
            selected = state.get("selected_specialists") or []
            round_ = state.get("coordinator_round", 1)
            # A more specific reason already set upstream (e.g.
            # business_agent.evaluate_read's "case_not_found_or_denied")
            # is preserved rather than overwritten by a generic one --
            # mirrors langgraph_experiment.nodes.make_synthesize_rag_node's
            # `state.get("termination_reason") or "insufficient_evidence"`
            # fallback pattern.
            reason = state.get("termination_reason") or (
                "max_rounds"
                if "knowledge" in selected and round_ >= MAX_COORDINATOR_ROUNDS
                else "insufficient_evidence"
            )
            return {
                "final_answer": (
                    "I don't have enough authorized, retrieved evidence to answer this "
                    "question confidently."
                ),
                "termination_reason": reason,
            }

        if deps.llm is None or deps.rag_prompt_template is None:
            parts: list[str] = []
            if knowledge_evidence:
                summary = "; ".join(f"{i['source']}: {i['content']}" for i in knowledge_evidence)
                parts.append(summary)
            if case_result is not None:
                parts.append(f"case {case_result['case_id']} is '{case_result['status']}'")
            return {
                "final_answer": " | ".join(parts) or "Synthesis dependencies unavailable.",
                "termination_reason": "synthesized",
            }

        context_blocks = [
            f"[Source {i}: {item['source']}]\n{item['content']}"
            for i, item in enumerate(knowledge_evidence, start=1)
        ]
        if case_result is not None:
            context_blocks.append(
                f"[Source case:{case_result['case_id']}]\n"
                f"Case {case_result['case_id']} "
                f"({case_result.get('subject', 'n/a')}) is currently "
                f"'{case_result['status']}', priority {case_result.get('priority', 'n/a')}, "
                f"assigned to {case_result.get('assigned_team', 'n/a')}. "
                f"{case_result.get('description', '')}"
            )
        context = "\n\n---\n\n".join(context_blocks)
        system, user = deps.rag_prompt_template.render(
            context=context, query=state["original_query"]
        )
        answer = deps.llm.generate(system, user)
        return {
            "final_answer": answer,
            "termination_reason": "synthesized",
            "llm_call_count": state.get("llm_call_count", 0) + 1,
        }

    return final_synthesis
