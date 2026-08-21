"""RAGAS-based scoring: faithfulness/relevancy/precision/recall/correctness/noise/factual.

A sibling module to `answer_quality.py`, not an extension of it.
`AnswerQualityScorer`'s single-scalar `score() -> float` contract doesn't
fit RAGAS's multi-metric, context-aware output. Optional: requires the
`ragas` extra (`pip install .[ragas]`); every `ragas`/`datasets` import is
lazy (inside functions) so importing this module never requires them.

Scores are produced by an LLM judge and have not been validated against
human labels. See `scripts/generate_manual_review.py` and
`scripts/compare_ragas_manual.py` before treating them as ground truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from rag.embedders.base import Embedder
from rag.generation.base import LLM

if TYPE_CHECKING:
    from ragas.evaluation import EvaluationResult

METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

# Metrics ragas exposes only as classes needing instantiation, not as
# ready-built singletons under `ragas.metrics` (unlike METRIC_NAMES above).
# Both use the same {user_input, response, retrieved_contexts, reference}
# columns `build_dataset` already produces, so no dataset changes are
# needed. Loaded best-effort in `_load_metrics`: an unsupported/renamed
# class in a future ragas version fails into `failed`, not a hard import
# error.
_EXTRA_METRIC_CLASSES = {
    "noise_sensitivity": "NoiseSensitivity",
    "factual_correctness": "FactualCorrectness",
}

CAVEAT = (
    "RAGAS scores are produced by an LLM judge and have not been validated "
    "against human labels in this environment. Do not treat them as ground "
    "truth for decision-making until reviewed. See "
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
    for name, class_name in _EXTRA_METRIC_CLASSES.items():
        try:
            metric_cls = getattr(ragas.metrics, class_name)
            metrics.append(metric_cls())
            used.append(name)
        except (AttributeError, TypeError) as exc:
            failed[name] = f"{type(exc).__name__}: {exc}"
    return metrics, used, failed


def _resolve_result_columns(used: list[str], df_columns: list[str]) -> dict[str, str | None]:
    """Map each intended metric name to its actual column in `ragas.evaluate()`'s result frame.

    Most metrics' result column is their bare name (e.g. `"faithfulness"`).
    But metrics with a `mode` parameter (`NoiseSensitivity`/
    `FactualCorrectness`, both added via `_EXTRA_METRIC_CLASSES`) come
    back as `"noise_sensitivity(mode=relevant)"`/
    `"factual_correctness(mode=f1)"` instead (confirmed directly against
    the pinned `ragas==0.3.9`'s `evaluate()` output, not assumed): ragas
    disambiguates columns by each metric instance's own config, not just
    its class name. Matching by prefix (`"<name>("`) tolerates that
    without hardcoding the exact mode string, which isn't part of this
    project's `_EXTRA_METRIC_CLASSES` config and could change if a metric
    class's default `mode` changes in a future ragas version.

    Parameters
    ----------
    used : list[str]
        Metric names this project intended to score (`METRIC_NAMES` plus
        any `_EXTRA_METRIC_CLASSES` keys that loaded successfully).
    df_columns : list[str]
        The actual columns of `result.to_pandas()`.

    Returns
    -------
    dict[str, str | None]
        `{intended_name: actual_column_name}`, or `None` for a name with
        no matching column at all (should not happen for a name in
        `used`, but handled defensively rather than raising `KeyError`).
    """
    resolved: dict[str, str | None] = {}
    for name in used:
        if name in df_columns:
            resolved[name] = name
            continue
        prefix = f"{name}("
        match = next((col for col in df_columns if col.startswith(prefix)), None)
        resolved[name] = match
    return resolved


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


def score(
    rows: list[dict[str, Any]],
    judge_llm: LLM,
    embedder: Embedder,
    cache: Any | None = None,
) -> dict[str, Any]:
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
    cache : Any | None, optional
        A `ragas.cache.CacheInterface` (e.g.
        `rag.eval.ragas_cache.NamespacedDiskCache`) memoizing judge calls
        across runs. `None` (the default) disables caching entirely --
        every call reaches the judge LLM, matching pre-caching behavior.
        Typed `Any` rather than `ragas.cache.CacheInterface` so importing
        this module never requires `ragas` to be installed.

    Returns
    -------
    dict[str, Any]
        ``{"metrics_used", "metrics_failed", "aggregate", "per_question",
        "caveat", "cache_stats"}``. ``cache_stats`` is ``None`` when
        `cache` is `None`, otherwise ``{"hits", "misses", "total"}``.

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
            "cache_stats": cache.stats.as_dict() if cache is not None else None,
        }

    dataset = build_dataset(rows)
    wrapped_llm = LangchainLLMWrapper(LangchainLLMAdapter(rag_llm=judge_llm), cache=cache)
    wrapped_embeddings = LangchainEmbeddingsWrapper(LangchainEmbeddingsAdapter(embedder))
    result = ragas_evaluate(
        dataset=dataset, metrics=metrics, llm=wrapped_llm, embeddings=wrapped_embeddings
    )
    # `ragas_evaluate` is typed to return `EvaluationResult | Executor`, but
    # `Executor` is only ever returned when `return_executor=True` is passed
    # (confirmed against the installed ragas's own `evaluate()` signature) --
    # never the case at this call site, so this is a real static fact, not a
    # blind assumption. `cast` (not a runtime isinstance check) since this
    # narrows a type-checker-only ambiguity without adding any runtime cost
    # or behavior change.
    df = cast("EvaluationResult", result).to_pandas()
    columns = _resolve_result_columns(used, list(df.columns))

    per_question = [
        {
            "question_index": row["question_index"],
            "question": row["question"],
            "unanswerable": row["unanswerable"],
            "scores": {
                m: (
                    float(df.iloc[i][columns[m]])
                    if columns[m] is not None and df.iloc[i][columns[m]] == df.iloc[i][columns[m]]
                    else None
                )
                for m in used
            },
        }
        for i, row in enumerate(rows)
    ]
    aggregate = {
        m: (float(df[columns[m]].mean()) if columns[m] is not None else None) for m in used
    }
    return {
        "metrics_used": used,
        "metrics_failed": failed,
        "aggregate": aggregate,
        "per_question": per_question,
        "cache_stats": cache.stats.as_dict() if cache is not None else None,
        "caveat": CAVEAT,
    }
