"""Deterministic evaluation harness for the multi-agent experiment.

Usage
-----
    python -m multi_agent_experiment.eval.run_multi_agent_eval
    python -m multi_agent_experiment.eval.run_multi_agent_eval --out report.json

Runs `multi_agent_gold.jsonl` (12 rows spanning knowledge-only,
business-only, mixed, business-mutation executed/rejected/invalid,
authorization-denial (same-tenant and cross-tenant), an
unnecessary-agent check, and a Knowledge Agent failure) against the
compiled graph, using the same self-contained synthetic knowledge base
and the real, unmodified `rag.mcp.business.store` that `cli/
scenario_walkthrough.py` uses -- **no external services needed** (no
Postgres, no Ollama), so this is runnable immediately and reproducibly.

Mirrors `rag.eval.run_agent_eval`'s spirit (deterministic, local-only
metrics, no LLM judge) at a fraction of its scope -- see that module for
production's own, much larger agentic-eval harness; this one exists
purely to answer README.md section 14's required metrics for this
specific experiment, not to replace it.

Metrics computed, matching README.md section 14's list:

- `routing_accuracy`: fraction of rows where
  `needs_knowledge`/`needs_business`/`is_case_mutation` all matched the
  gold-expected routing decision.
- `unnecessary_agent_rate`: fraction of rows where a specialist ran that
  the gold row did *not* expect (independent of `routing_accuracy`'s
  all-or-nothing scoring, so a single spuriously-invoked specialist is
  visible even on an otherwise-correct row).
- `tool_call_count`/`llm_call_count`/`latency_ms`: per-row and mean.
- `answer_correctness`: fraction of rows where every
  `expected_answer_keywords` string appears in the final answer
  (case-insensitive substring match -- the same
  `KeywordOverlapScorer`-style heuristic `rag.eval.answer_quality` uses,
  not an LLM judge).
- `citation_grounding`: fraction of rows with an `expected_citation_count`
  whose actual citation count matched.
- `security_failures`: count of rows where a `forbidden_answer_substrings`
  entry (case-insensitive) leaked into the final answer -- this must
  always be `0`; a non-zero value is a real authorization/redaction bug,
  not a quality nit.
- `termination_reasons`: a breakdown, for diagnosing *why* a row failed,
  not just that it did.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langgraph_experiment.wiring import GraphDeps
from multi_agent_experiment.graph import build_graph
from rag.config import load_config

_GOLD_PATH = Path(__file__).resolve().parent / "multi_agent_gold.jsonl"


class _FakeSearchResult:
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
    _FakeSearchResult("kb-3", "docs/sla.md", "The support SLA response window is 4 hours.", 0.9),
]


class _FakePipeline:
    def retrieve(self, query, *, filters=None, candidate_k=None, auth=None):
        lowered = query.lower()
        return [
            result
            for result in _SYNTHETIC_KNOWLEDGE_BASE
            if any(
                word in lowered for word in result.chunk.content.lower().split() if len(word) > 4
            )
        ]


class _FailingPipeline:
    def retrieve(self, query, *, filters=None, candidate_k=None, auth=None):
        raise TimeoutError("simulated knowledge-base failure")


class _FakeLLM:
    def generate(self, system: str, user: str) -> str:
        return f"grounded answer based on: {user}"

    def health_check(self) -> bool:
        return True


class _FakePromptTemplate:
    def render(self, **kwargs):
        return ("system prompt", f"query={kwargs.get('query')}\ncontext={kwargs.get('context')}")


def _load_gold() -> list[dict]:
    rows = []
    with _GOLD_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_deps(pipeline) -> GraphDeps:
    return GraphDeps(
        config=load_config(),
        pipeline=pipeline,
        llm=_FakeLLM(),
        rag_prompt_template=_FakePromptTemplate(),
        approval_roles=("case_status_approver",),
        cross_tenant_support_roles=("techfusion_support",),
    )


def _run_one(row: dict) -> dict[str, Any]:
    pipeline = _FailingPipeline() if row.get("simulate_knowledge_failure") else _FakePipeline()
    graph = build_graph(_build_deps(pipeline), InMemorySaver())
    config = {"configurable": {"thread_id": f"eval-{row['id']}"}}

    started_at = time.perf_counter()
    result = graph.invoke(
        {
            "original_query": row["query"],
            "tenant_id": row.get("tenant_id"),
            "roles": row.get("roles") or [],
        },
        config,
    )
    if "__interrupt__" in result:
        if row.get("auto_approve"):
            result = graph.invoke(
                Command(resume={"decision": "approve", "approver_roles": ["case_status_approver"]}),
                config,
            )
        elif row.get("reject"):
            result = graph.invoke(Command(resume={"decision": "reject"}), config)
    latency_ms = (time.perf_counter() - started_at) * 1000

    answer = (result.get("final_answer") or "").lower()
    keyword_hits = [kw for kw in row.get("expected_answer_keywords", []) if kw.lower() in answer]
    keywords_ok = len(keyword_hits) == len(row.get("expected_answer_keywords", []))
    leaked = [s for s in row.get("forbidden_answer_substrings", []) if s.lower() in answer]

    routing_ok = (
        result.get("needs_knowledge", False) == row.get("expected_needs_knowledge", False)
        and result.get("needs_business", False) == row.get("expected_needs_business", False)
        and result.get("is_case_mutation", False) == row.get("expected_is_mutation", False)
    )
    unnecessary_knowledge = result.get("needs_knowledge", False) and not row.get(
        "expected_needs_knowledge", False
    )
    unnecessary_business = result.get("needs_business", False) and not row.get(
        "expected_needs_business", False
    )
    termination_ok = (
        row.get("expected_termination_reason") is None
        or result.get("termination_reason") == row["expected_termination_reason"]
    )
    citation_count = len(result.get("citations") or [])
    citation_ok = (
        "expected_citation_count" not in row or citation_count == row["expected_citation_count"]
    )

    return {
        "id": row["id"],
        "category": row["category"],
        "routing_ok": routing_ok,
        "unnecessary_agent": unnecessary_knowledge or unnecessary_business,
        "termination_ok": termination_ok,
        "termination_reason": result.get("termination_reason"),
        "answer_ok": keywords_ok and not leaked,
        "leaked_substrings": leaked,
        "citation_ok": citation_ok,
        "citation_count": citation_count,
        "tool_call_count": len(result.get("tool_call_log") or []),
        "llm_call_count": result.get("llm_call_count", 0),
        "latency_ms": latency_ms,
    }


def run() -> dict[str, Any]:
    """Run every gold row and return the aggregate report dict."""
    rows = _load_gold()
    per_example = [_run_one(row) for row in rows]

    n = len(per_example)
    security_failures = sum(1 for r in per_example if r["leaked_substrings"])
    report = {
        "num_examples": n,
        "routing_accuracy": sum(r["routing_ok"] for r in per_example) / n,
        "unnecessary_agent_rate": sum(r["unnecessary_agent"] for r in per_example) / n,
        "termination_reason_accuracy": sum(r["termination_ok"] for r in per_example) / n,
        "answer_correctness": sum(r["answer_ok"] for r in per_example) / n,
        "citation_grounding": sum(r["citation_ok"] for r in per_example) / n,
        "security_failures": security_failures,
        "mean_tool_call_count": sum(r["tool_call_count"] for r in per_example) / n,
        "mean_llm_call_count": sum(r["llm_call_count"] for r in per_example) / n,
        "mean_latency_ms": sum(r["latency_ms"] for r in per_example) / n,
        "termination_reasons": dict(Counter(r["termination_reason"] for r in per_example)),
        "per_example": per_example,
    }
    return report


def _print_report(report: dict) -> None:
    print(f"num_examples: {report['num_examples']}")
    for key in (
        "routing_accuracy",
        "unnecessary_agent_rate",
        "termination_reason_accuracy",
        "answer_correctness",
        "citation_grounding",
        "mean_tool_call_count",
        "mean_llm_call_count",
        "mean_latency_ms",
    ):
        print(f"{key}: {report[key]:.3f}")
    print(f"security_failures: {report['security_failures']}  (must be 0)")
    print(f"termination_reasons: {report['termination_reasons']}")
    failures = [
        r
        for r in report["per_example"]
        if not (r["routing_ok"] and r["answer_ok"] and r["termination_ok"] and r["citation_ok"])
    ]
    if failures:
        print("\nfailing rows:")
        for r in failures:
            print(f"  - {r['id']} ({r['category']}): {r}")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: run the gold set, print the report, optionally write it to `--out`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="Optional path to write the full JSON report")
    args = parser.parse_args(argv)

    report = run()
    _print_report(report)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nFull report written to {args.out}")
    return 0 if report["security_failures"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
