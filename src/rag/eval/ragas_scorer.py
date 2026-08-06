"""RAGAS-based generation-quality scoring: faithfulness/relevancy/precision/recall/correctness.

A sibling module to `answer_quality.py`, not an extension of it —
`AnswerQualityScorer`'s single-scalar `score() -> float` contract doesn't
fit RAGAS's multi-metric, context-aware output. Optional: requires the
`ragas` extra (`pip install .[ragas]`); every `ragas`/`datasets` import is
lazy (inside functions) so importing this module never requires them.

Scores are produced by an LLM judge and have not been validated against
human labels — see `scripts/generate_manual_review.py` and
`scripts/compare_ragas_manual.py` before treating them as ground truth.
"""

from __future__ import annotations

from typing import Any

from rag.embedders.base import Embedder
from rag.generation.base import LLM

METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

CAVEAT = (
    "RAGAS scores are produced by an LLM judge and have not been validated "
    "against human labels in this environment. Do not treat them as ground "
    "truth for decision-making until reviewed — see "
    "scripts/generate_manual_review.py and scripts/compare_ragas_manual.py."
)


def _load_metrics() -> tuple[list[Any], list[str], dict[str, str]]:
    """Best-effort import each of `METRIC_NAMES` from `ragas.metrics`.

    One metric being unavailable in the installed ragas version doesn't
    block the others (e.g. `answer_correctness` per the "if supported
    cleanly" requirement).

    Returns
    -------
    tuple[list[Any], list[str], dict[str, str]]
        ``(metric objects, names successfully imported, {name: error} for
        the ones that weren't)``.
    """
    import ragas.metrics

    metrics: list[Any] = []
    used: list[str] = []
    failed: dict[str, str] = {}
    for name in METRIC_NAMES:
        try:
            metrics.append(getattr(ragas.metrics, name))
            used.append(name)
        except AttributeError as exc:
            failed[name] = f"{type(exc).__name__}: {exc}"
    return metrics, used, failed


def build_dataset(rows: list[dict[str, Any]]) -> Any:
    """Build a ragas-shaped `datasets.Dataset` from scoring rows.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Each dict must have ``question``, ``answer``, ``contexts``
        (``list[str]``), and ``expected_answer``.

    Returns
    -------
    datasets.Dataset
        Built with the ``user_input``/``response``/``retrieved_contexts``/
        ``reference`` column names ragas has used since v0.2.

    Raises
    ------
    RuntimeError
        If the ``datasets`` package isn't installed.
    """
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS scoring requires the 'ragas' extra (includes datasets). "
            "Install with: pip install .[ragas]"
        ) from exc
    return Dataset.from_dict(
        {
            "user_input": [r["question"] for r in rows],
            "response": [r["answer"] for r in rows],
            "retrieved_contexts": [r["contexts"] for r in rows],
            "reference": [r["expected_answer"] for r in rows],
        }
    )


def score(rows: list[dict[str, Any]], judge_llm: LLM, embedder: Embedder) -> dict[str, Any]:
    """Run RAGAS `evaluate()` over `rows` using `judge_llm` and `embedder`.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        One dict per scoreable example: ``question_index``, ``question``,
        ``unanswerable``, ``answer``, ``contexts`` (``list[str]``),
        ``expected_answer``.
    judge_llm : LLM
        This project's `LLM` ABC instance used as the RAGAS judge.
    embedder : Embedder
        This project's `Embedder` ABC instance for embedding-based metrics.

    Returns
    -------
    dict[str, Any]
        ``{"metrics_used", "metrics_failed", "aggregate", "per_question",
        "caveat"}``.

    Raises
    ------
    RuntimeError
        If the ``ragas`` package isn't installed.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS scoring requires the 'ragas' extra. Install with: pip install .[ragas]"
        ) from exc
    from rag.eval.ragas_adapters import LangchainEmbeddingsAdapter, LangchainLLMAdapter

    metrics, used, failed = _load_metrics()
    if not rows or not metrics:
        return {
            "metrics_used": used,
            "metrics_failed": failed,
            "aggregate": {},
            "per_question": [],
            "caveat": CAVEAT,
        }

    dataset = build_dataset(rows)
    wrapped_llm = LangchainLLMWrapper(LangchainLLMAdapter(rag_llm=judge_llm))
    wrapped_embeddings = LangchainEmbeddingsWrapper(LangchainEmbeddingsAdapter(embedder))
    result = ragas_evaluate(
        dataset=dataset, metrics=metrics, llm=wrapped_llm, embeddings=wrapped_embeddings
    )
    df = result.to_pandas()

    per_question = [
        {
            "question_index": row["question_index"],
            "question": row["question"],
            "unanswerable": row["unanswerable"],
            "scores": {
                m: (
                    float(df.iloc[i][m])
                    if m in df.columns and df.iloc[i][m] == df.iloc[i][m]
                    else None
                )
                for m in used
            },
        }
        for i, row in enumerate(rows)
    ]
    aggregate = {m: (float(df[m].mean()) if m in df.columns else None) for m in used}
    return {
        "metrics_used": used,
        "metrics_failed": failed,
        "aggregate": aggregate,
        "per_question": per_question,
        "caveat": CAVEAT,
    }
