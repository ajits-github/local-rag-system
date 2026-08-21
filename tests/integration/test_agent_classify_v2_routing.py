"""Real-LLM regression test for agent_classify_v2's simple/complex boundary fix.

`experiment_029` (see experiments/reports/agentic_rag_baseline_v1.md
section 3/4) found `unnecessary_agent_rate=1.0`: `agent_classify_v1`
routed both gold `tool_not_needed` questions -- single-document factual
lookups -- through the expensive agent path. That fix lives entirely in
prompt wording (`agent_classify_v2.yaml`), not in `rag.agent.graph`'s
control-flow code, so it can only be regression-tested against a real
LLM, not a scripted mock (see `tests/unit/test_agent_graph_routing.py`
for the mocked, deterministic proof that the graph correctly follows
whatever classification it's given).

Uses paraphrased, gold-analogous questions -- not the gold file's own
question text -- so a pass here is evidence the fix generalizes, not that
the model memorized the benchmark.
"""

from __future__ import annotations

import pytest

from rag.agent.decisions import ClassifyDecision, run_decision
from rag.config import REPO_ROOT
from rag.factory import build_llm
from rag.prompts.loader import load_prompt_template

_TEMPLATE_PATH = REPO_ROOT / "src/rag/prompts/templates/agent_classify_v2.yaml"

_SIMPLE_QUESTIONS = [
    "What is the maximum number of retries allowed for a failed API call?",
    "How many hours after account cancellation is customer data purged?",
]

_COMPLEX_QUESTIONS = [
    "Which team owns the service that failed during last week's outage, and "
    "what is that team's on-call escalation path?",
    "What is the currently effective API rate limit for partner integrations, "
    "and what happens if a client's request is not backward compatible?",
]


def _classify(question: str, llm) -> str:
    template = load_prompt_template(_TEMPLATE_PATH)
    decision = run_decision(llm, template, ClassifyDecision, 1, query=question)
    assert decision is not None, f"classify decision failed to parse for: {question!r}"
    return decision.query_type


@pytest.fixture
def classify_llm(require_ollama, config):
    """Build a real Ollama LLM (qwen2.5:3b, matching the agentic-rag baseline configs)."""
    agentic = config.model_copy(deep=True)
    agentic.generation.model_name = "qwen2.5:3b"
    return build_llm(agentic)


@pytest.mark.parametrize("question", _SIMPLE_QUESTIONS)
def test_single_document_factual_questions_classify_simple(classify_llm, question):
    """A direct, single-document fact stays "simple" even though it needs a targeted search."""
    assert _classify(question, classify_llm) == "simple"


@pytest.mark.parametrize("question", _COMPLEX_QUESTIONS)
def test_multi_hop_questions_classify_complex(classify_llm, question):
    """A question needing two dependent, separately-sourced facts still classifies "complex"."""
    assert _classify(question, classify_llm) == "complex"
