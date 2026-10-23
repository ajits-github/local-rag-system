"""Deterministic ontology-aware-vs-ordinary GraphRAG evaluation.

No LLM judge and no RAGAS here, deliberately, matching
`rag.eval.run_graph_eval`'s own precedent for the GraphRAG milestone's
own small gold set: every question in `data/eval/ontology_gold.jsonl`
has an exact, deterministic expected answer (an entity set, a
canonical class name, or a resolution status), so scoring it against
an LLM-judge pipeline would add cost and non-determinism for no real
benefit. See `docs/ontology.md` section 18 for the full design and the
last recorded run's numbers.

Reports, per question: retrieval recall (and, for `type_expansion`
questions only, BOTH a naive and an ontology-aware recall, which is
the actual "ordinary GraphRAG vs. ontology-aware GraphRAG" comparison),
exact-match "answer correctness", and latency. Aggregates:
`retrieval_recall` (naive vs ontology-aware, `type_expansion` rows
only), `answer_correctness` (all rows), `semantic_mapping_accuracy`
(`class_mapping` + `entity_resolution` rows), `graph_path_correctness`
(`sparql_query` rows), `latency_ms` (mean, naive vs ontology-aware
separately where both were run).

Usage:
    python -m rag.eval.run_ontology_eval
    python -m rag.eval.run_ontology_eval --gold data/eval/ontology_gold.jsonl \
        --output /path/to/report.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

from rag.config import REPO_ROOT
from rag.ontology import paths
from rag.ontology.graphrag_integration import naive_type_match, ontology_aware_type_match
from rag.ontology.interoperability import (
    find_equivalent_external_classes,
    summarize_entity_resolution,
)
from rag.ontology.sparql_runner import run_query_file
from rag.ontology.store import load_store

ONTOLOGY_NS = "http://techfusion.example/ontology/v1#"


def _local_name(iri: str) -> str:
    """Shorten a full IRI to its final path/fragment segment for comparison against gold data."""
    for separator in ("#", "/"):
        if separator in iri:
            iri = iri.rsplit(separator, 1)[-1]
    return iri


def _recall(expected: set[str], actual: set[str]) -> float:
    """Compute recall; an unanswerable (empty `expected`) scores 1.0 iff `actual` is empty too."""
    if not expected:
        return 1.0 if not actual else 0.0
    return len(expected & actual) / len(expected)


def _score_type_expansion(row: dict, data_store, ontology_store) -> dict:
    """Score a `type_expansion` gold row: naive vs ontology-aware type match."""
    class_iri = ONTOLOGY_NS + row["query_class"]
    expected = set(row["expected_entities"])

    naive_start = time.perf_counter()
    naive = naive_type_match(data_store, class_iri)
    naive_ms = (time.perf_counter() - naive_start) * 1000

    aware_start = time.perf_counter()
    aware = ontology_aware_type_match(data_store, ontology_store, class_iri)
    aware_ms = (time.perf_counter() - aware_start) * 1000

    naive_actual = {_local_name(e) for e in naive.entities}
    aware_actual = {_local_name(e) for e in aware.entities}

    return {
        "naive_recall": _recall(expected, naive_actual),
        "ontology_aware_recall": _recall(expected, aware_actual),
        "naive_correct": naive_actual == expected,
        "ontology_aware_correct": aware_actual == expected,
        "naive_latency_ms": naive_ms,
        "ontology_aware_latency_ms": aware_ms,
        "naive_entities": sorted(naive_actual),
        "ontology_aware_entities": sorted(aware_actual),
        "expanded_classes": [_local_name(c) for c in aware.expanded_classes],
    }


def _score_sparql_query(row: dict, data_store) -> dict:
    """Score a `sparql_query` gold row: run the referenced `.rq` file and check its result set."""
    expected = set(row["expected_entities"])
    query_path = paths.SPARQL_DIR / row["sparql_query_file"]

    start = time.perf_counter()
    rows = run_query_file(data_store, query_path)
    latency_ms = (time.perf_counter() - start) * 1000

    var_name = row["result_variable"]
    actual = {_local_name(value) for r in rows if (value := r.get(var_name)) is not None}

    return {
        "recall": _recall(expected, actual),
        "correct": actual == expected,
        "latency_ms": latency_ms,
        "actual_entities": sorted(actual),
    }


def _score_class_mapping(row: dict, data_store) -> dict:
    """Score a `class_mapping` gold row against `owl:equivalentClass` mappings."""
    start = time.perf_counter()
    mappings = find_equivalent_external_classes(data_store)
    latency_ms = (time.perf_counter() - start) * 1000

    mapped_canonical_names = {
        _local_name(m.canonical_class)
        for m in mappings
        if _local_name(m.external_class) in set(row["external_classes"])
    }
    correct = mapped_canonical_names == {row["expected_canonical_class"]}

    return {
        "correct": correct,
        "resolved_canonical_classes": sorted(mapped_canonical_names),
        "latency_ms": latency_ms,
    }


def _score_entity_resolution(row: dict, data_store) -> dict:
    """Score an `entity_resolution` gold row: confirmed, proposed, or unresolved."""
    start = time.perf_counter()
    links = summarize_entity_resolution(data_store)
    latency_ms = (time.perf_counter() - start) * 1000

    matching = [link for link in links if row["external_identifier"] in link.external_entity]
    expected = row["expected_resolution"]

    if any(link.confirmed for link in matching):
        actual_resolution = "confirmed"
    elif any(not link.confirmed for link in matching):
        actual_resolution = "proposed"
    else:
        actual_resolution = "unresolved"

    return {
        "correct": actual_resolution == expected,
        "actual_resolution": actual_resolution,
        "latency_ms": latency_ms,
    }


_SCORERS = {
    "type_expansion": lambda row, data_store, ontology_store: _score_type_expansion(
        row, data_store, ontology_store
    ),
    "sparql_query": lambda row, data_store, ontology_store: _score_sparql_query(row, data_store),
    "class_mapping": lambda row, data_store, ontology_store: _score_class_mapping(row, data_store),
    "entity_resolution": lambda row, data_store, ontology_store: _score_entity_resolution(
        row, data_store
    ),
}


def run(gold_path: Path) -> dict:
    """Run every gold question and return the full report dict.

    Parameters
    ----------
    gold_path : Path
        Path to a JSONL gold file in `data/eval/ontology_gold.jsonl`'s schema.

    Returns
    -------
    dict
        `{"per_question": [...], "summary": {...}}`.
    """
    data_store = load_store(paths.ALL_VALID_FILES)
    ontology_store = (
        data_store  # ontology + instances loaded together; expansion only reads T-box triples
    )

    rows = [
        json.loads(line)
        for line in gold_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    per_question = []
    for row in rows:
        method = row["method"]
        if method not in _SCORERS:
            raise ValueError(f"Unknown eval method: {method!r} in question {row['id']!r}")
        scored = _SCORERS[method](row, data_store, ontology_store)

        per_question.append(
            {
                "id": row["id"],
                "question": row["question"],
                "category": row["category"],
                "method": method,
                "requires_ontology_awareness": row.get("requires_ontology_awareness", False),
                **scored,
            }
        )

    return {"per_question": per_question, "summary": _summarize(per_question)}


def _summarize(per_question: list[dict]) -> dict:
    """Aggregate per-question scores into the report's `summary` block."""
    type_expansion_rows = [r for r in per_question if r["method"] == "type_expansion"]
    sparql_rows = [r for r in per_question if r["method"] == "sparql_query"]
    mapping_rows = [
        r for r in per_question if r["method"] in ("class_mapping", "entity_resolution")
    ]

    all_correct = [
        r.get("correct", r.get("ontology_aware_correct"))
        for r in per_question
        if "correct" in r or "ontology_aware_correct" in r
    ]

    summary: dict[str, object] = {
        "question_count": len(per_question),
        "answer_correctness": mean(bool(c) for c in all_correct) if all_correct else None,
    }

    if type_expansion_rows:
        summary["retrieval_recall"] = {
            "naive": mean(r["naive_recall"] for r in type_expansion_rows),
            "ontology_aware": mean(r["ontology_aware_recall"] for r in type_expansion_rows),
        }
        summary["latency_ms"] = {
            "naive_mean": mean(r["naive_latency_ms"] for r in type_expansion_rows),
            "ontology_aware_mean": mean(
                r["ontology_aware_latency_ms"] for r in type_expansion_rows
            ),
        }
        improved = [r for r in type_expansion_rows if r["requires_ontology_awareness"]]
        unchanged = [r for r in type_expansion_rows if not r["requires_ontology_awareness"]]
        summary["ontology_awareness_impact"] = {
            "rows_where_expansion_should_help": len(improved),
            "rows_where_expansion_should_not_change_result": len(unchanged),
            "recall_delta_on_rows_expected_to_improve": (
                mean(r["ontology_aware_recall"] - r["naive_recall"] for r in improved)
                if improved
                else None
            ),
            "recall_delta_on_rows_expected_unchanged": (
                mean(r["ontology_aware_recall"] - r["naive_recall"] for r in unchanged)
                if unchanged
                else None
            ),
        }

    if sparql_rows:
        summary["graph_path_correctness"] = mean(bool(r["correct"]) for r in sparql_rows)
        summary["sparql_recall"] = mean(r["recall"] for r in sparql_rows)

    if mapping_rows:
        summary["semantic_mapping_accuracy"] = mean(bool(r["correct"]) for r in mapping_rows)

    return summary


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=str(REPO_ROOT / "data" / "eval" / "ontology_gold.jsonl"))
    parser.add_argument("--output", default=None, help="Write the full JSON report here too")
    args = parser.parse_args()

    report = run(Path(args.gold))
    print(json.dumps(report, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
