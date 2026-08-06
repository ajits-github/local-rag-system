"""Validate a gold JSONL file before running evaluation against it.

Loaded like the other scripts/*.py modules (not a package) via a sys.path
insert onto src/, matching init_db.py/record_experiment.py's pattern.

Checks, in order:
  1. record count vs --expect-count (warning only -- the file may grow).
  2. every relevant_documents entry resolves to a real file (hard fail).
  3. answerable questions' expected_answer keyword-overlaps its referenced
     content (warning only -- KeywordOverlapScorer is a crude heuristic,
     see eval/answer_quality.py's own documented caveat).
  4. every structured-content bucket (table/code_configuration/chart) has
     at least one example (hard fail -- proves buckets are "clearly
     identifiable").
  5. no duplicate questions, case-folded exact match (hard fail).

Exit code is non-zero iff checks 2, 4, or 5 find a problem.

Usage:
    python scripts/validate_gold_file.py --gold data/eval/techfusion_gold.jsonl \
        --kb-root data/knowledge_base --expect-count 62
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.cleaners.default_cleaner import DefaultCleaner  # noqa: E402
from rag.config import DEFAULT_CONFIG_PATH, AppConfig, load_config  # noqa: E402
from rag.eval.answer_quality import KeywordOverlapScorer  # noqa: E402
from rag.eval.content_type import build_document_content_types, classify_example  # noqa: E402
from rag.eval.gold_schema import GoldExample, load_gold_jsonl  # noqa: E402
from rag.loaders.registry import get_loader  # noqa: E402

_STRUCTURED_BUCKETS = ("table", "code_configuration", "chart")
_KEYWORD_OVERLAP_THRESHOLD = 0.15


def _load_document_text(data_root: Path, relative_path: str) -> str | None:
    """Load and clean one `relevant_documents` entry's full text, or None if missing."""
    file_path = data_root / relative_path
    if not file_path.is_file():
        return None
    raw = get_loader(file_path).load(file_path)
    return DefaultCleaner().clean(raw.content)


def validate(
    examples: list[GoldExample],
    data_root: Path,
    config: AppConfig,
    expect_count: int | None,
) -> tuple[list[str], list[str]]:
    """Run every check and split findings into (errors, warnings).

    Parameters
    ----------
    examples : list[GoldExample]
        Parsed gold file rows.
    data_root : Path
        Root that each `relevant_documents` entry is relative to -- the
        parent of the knowledge-base directory, so a gold entry like
        "knowledge_base/architecture/x.md" resolves to a real file.
    config : AppConfig
        Used to build the content-type classifier's chunker chain.
    expect_count : int | None
        Expected record count, or None to skip that check.

    Returns
    -------
    tuple[list[str], list[str]]
        `(errors, warnings)`. Errors are the hard failures (missing
        documents, empty structured buckets, duplicate questions);
        warnings never affect the exit code.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if expect_count is not None and len(examples) != expect_count:
        warnings.append(f"Expected {expect_count} records, found {len(examples)}.")

    document_texts: dict[str, str] = {}
    for example in examples:
        for relative_path in example.relevant_documents:
            if relative_path in document_texts:
                continue
            text = _load_document_text(data_root, relative_path)
            if text is None:
                errors.append(
                    f"relevant_documents entry does not resolve to a file: "
                    f"'{relative_path}' (question: {example.question!r})"
                )
            else:
                document_texts[relative_path] = text

    scorer = KeywordOverlapScorer()
    for example in examples:
        if example.unanswerable or not example.expected_answer:
            continue
        combined = "\n".join(
            document_texts[p] for p in example.relevant_documents if p in document_texts
        )
        if not combined:
            continue
        score = scorer.score(example.question, combined, example.expected_answer)
        if score < _KEYWORD_OVERLAP_THRESHOLD:
            warnings.append(
                f"Low keyword overlap ({score:.2f}) between expected_answer and referenced "
                f"content: {example.question!r}"
            )

    document_content_types = build_document_content_types(
        data_root, (p for e in examples for p in e.relevant_documents), config
    )
    bucket_counts = Counter(classify_example(e, document_content_types) for e in examples)
    for bucket in _STRUCTURED_BUCKETS:
        if bucket_counts.get(bucket, 0) == 0:
            errors.append(f"No gold examples classified into the '{bucket}' bucket.")

    seen: Counter[str] = Counter(e.question.strip().casefold() for e in examples)
    for question, count in seen.items():
        if count > 1:
            errors.append(f"Duplicate question ({count}x): {question!r}")

    return errors, warnings


def main() -> None:
    """CLI entrypoint: validate a gold file and print a pass/fail report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="Path to a gold JSONL file")
    parser.add_argument(
        "--kb-root",
        required=True,
        help="Knowledge-base directory that relevant_documents entries are relative to",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--expect-count", type=int, default=None)
    args = parser.parse_args()

    examples = load_gold_jsonl(Path(args.gold))
    config = load_config(args.config)
    data_root = Path(args.kb_root).parent
    errors, warnings = validate(examples, data_root, config, args.expect_count)

    print(f"Validated {len(examples)} records from {args.gold}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")

    if errors:
        print(f"\nFAILED: {len(errors)} error(s), {len(warnings)} warning(s).")
        raise SystemExit(1)
    print(f"\nPASSED: {len(warnings)} warning(s).")


if __name__ == "__main__":
    main()
