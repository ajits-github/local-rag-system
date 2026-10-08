"""Run all five required scenarios (README.md section 7) in one process and print a trace of each.

Usage
-----
    python -m multi_agent_experiment.cli.scenario_walkthrough

Uses a small, self-contained fake `RetrievalPipeline`/`LLM` (a synthetic
knowledge-base stand-in, clearly labeled below) so this script runs with
**no external services at all** (no Postgres, no Ollama), and always
produces the same, reproducible trace. This is deliberately a different
tradeoff from `run_demo.py`/`compare_harness.py`, both of which talk to
the real `RetrievalPipeline` (needing `make up` + native Ollama).
this script exists specifically to be the "deliverables" walkthrough
(README.md section 17's "example traces" requirement) that anyone can run
immediately, with zero setup, and get the exact five traces documented in
README.md. The business branch throughout uses the real, unmodified
`rag.mcp.business.store` (in-memory, no external service needed either
way).
"""

from __future__ import annotations

import json

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langgraph_experiment.wiring import GraphDeps
from multi_agent_experiment.graph import build_graph
from rag.config import load_config


class _FakeSearchResult:
    """Enough of a `rag.schemas.SearchResult` for the Knowledge Agent's own field access."""

    def __init__(self, chunk_id: str, source: str, content: str, score: float) -> None:
        self.chunk = _FakeChunk(chunk_id, source, content)
        self.score = score


class _FakeChunk:
    def __init__(self, chunk_id: str, source: str, content: str) -> None:
        self.content = content
        self.metadata = _FakeMetadata(chunk_id, source)


class _FakeMetadata:
    def __init__(self, chunk_id: str, source: str) -> None:
        self.chunk_id = chunk_id
        self.source = source


_SYNTHETIC_KNOWLEDGE_BASE = [
    _FakeSearchResult("kb-1", "docs/api-policy.md", "The API request timeout is 30 seconds.", 0.91),
    _FakeSearchResult(
        "kb-2",
        "docs/support-policy.md",
        "When a case is resolved, support should confirm with the customer before closing it.",
        0.88,
    ),
]


class _FakePipeline:
    """A synthetic knowledge base: returns only chunks whose own keyword matches the query."""

    def retrieve(self, query, *, filters=None, candidate_k=None, auth=None):
        lowered = query.lower()
        matches = [
            result
            for result in _SYNTHETIC_KNOWLEDGE_BASE
            if any(
                word in lowered for word in result.chunk.content.lower().split() if len(word) > 4
            )
        ]
        return matches


class _TimeoutPipeline:
    """Simulates a Knowledge Agent tool failure (CASE 5)."""

    def retrieve(self, query, *, filters=None, candidate_k=None, auth=None):
        raise TimeoutError("simulated knowledge-base timeout")


class _FakeLLM:
    def generate(self, system: str, user: str) -> str:
        return f"[synthesized answer grounded in]: {user[:200]}..."

    def health_check(self) -> bool:
        return True


class _FakePromptTemplate:
    def render(self, **kwargs):
        return ("system prompt", f"query={kwargs.get('query')}\ncontext={kwargs.get('context')}")


def _print_trace(title: str, result: dict) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    if "__interrupt__" in result:
        print("status: PAUSED (awaiting approval)")
        print(json.dumps(result["__interrupt__"][0].value, indent=2, default=str))
        return
    print(f"needs_knowledge={result.get('needs_knowledge')}")
    print(f"needs_business={result.get('needs_business')}")
    print(f"selected_specialists={result.get('selected_specialists')}")
    print(f"termination_reason={result.get('termination_reason')}")
    print(f"answer: {result.get('final_answer')}")
    if result.get("citations"):
        print(f"citations: {result['citations']}")
    if result.get("critic_notes"):
        print(f"critic_notes: {result['critic_notes']}")


def _deps(pipeline) -> GraphDeps:
    return GraphDeps(
        config=load_config(),
        pipeline=pipeline,
        llm=_FakeLLM(),
        rag_prompt_template=_FakePromptTemplate(),
        approval_roles=("case_status_approver",),
        cross_tenant_support_roles=("techfusion_support",),
    )


def main() -> int:
    """Run all five required scenarios in sequence and print a trace of each."""
    # CASE 1: pure knowledge question.
    graph = build_graph(_deps(_FakePipeline()), InMemorySaver())
    result = graph.invoke(
        {"original_query": "What is the current API timeout policy?"},
        {"configurable": {"thread_id": "case1"}},
    )
    _print_trace("CASE 1: pure knowledge question", result)

    # CASE 2: pure business question.
    result = graph.invoke(
        {
            "original_query": "What is the status of CASE-1001?",
            "tenant_id": "tenant_alpha",
            "roles": ["tenant_alpha_operator"],
        },
        {"configurable": {"thread_id": "case2"}},
    )
    _print_trace("CASE 2: pure business question", result)

    # CASE 3: mixed question, parallel dispatch.
    result = graph.invoke(
        {
            "original_query": (
                "According to the support policy, what should happen next for CASE-1001?"
            ),
            "tenant_id": "tenant_alpha",
            "roles": ["tenant_alpha_operator"],
        },
        {"configurable": {"thread_id": "case3"}},
    )
    _print_trace("CASE 3: mixed question (parallel Knowledge + Business)", result)

    # CASE 4: business mutation -> pause -> approve.
    paused = graph.invoke(
        {
            "original_query": "Please close CASE-2001",
            "tenant_id": "tenant_beta",
            "roles": ["tenant_beta_operator"],
        },
        {"configurable": {"thread_id": "case4"}},
    )
    _print_trace("CASE 4a: business mutation, paused for approval", paused)
    approved = graph.invoke(
        Command(resume={"decision": "approve", "approver_roles": ["case_status_approver"]}),
        {"configurable": {"thread_id": "case4"}},
    )
    _print_trace("CASE 4b: business mutation, approved and executed", approved)

    # CASE 5: Knowledge Agent failure -> bounded retry -> honest "insufficient evidence."
    failing_graph = build_graph(_deps(_TimeoutPipeline()), InMemorySaver())
    result = failing_graph.invoke(
        {"original_query": "What is the current API timeout policy?"},
        {"configurable": {"thread_id": "case5"}},
    )
    _print_trace("CASE 5: Knowledge Agent fails -> bounded retry -> no fabrication", result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
