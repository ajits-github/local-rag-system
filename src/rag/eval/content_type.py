"""Buckets gold questions and KB documents by structured-content type.

Document content types are derived by running the *configured* chunker
over each referenced file (self-updating. A newly added table/code/chart
document is picked up the next time this runs, not a hardcoded lookup
table), reusing the same loader -> cleaner -> chunker chain
`IngestionPipeline.ingest_file` runs, minus embed/write.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from rag.chunkers.registry import get_chunker
from rag.cleaners.default_cleaner import DefaultCleaner
from rag.config import AppConfig
from rag.eval.gold_schema import GoldExample
from rag.loaders.registry import get_loader

_CHART_KEYWORDS = ("chart", "caption", "trend")

_CONTENT_TYPE_TO_BUCKET = {
    "table": "table",
    "code": "code_configuration",
    "configuration": "code_configuration",
    "chart": "chart",
}
# Deterministic tie-break when a single document mixes more than one
# structured-content bucket (two real TechFusion documents contain both a
# table and a chart) and the keyword heuristic in `classify_example`
# doesn't resolve it.
_BUCKET_PRIORITY = ("table", "code_configuration", "chart")


def build_document_content_types(
    data_root: Path, relevant_paths: Iterable[str], config: AppConfig
) -> dict[str, set[str]]:
    """Run the configured chunker over each referenced KB file and collect its content types.

    Parameters
    ----------
    data_root : Path
        Root directory that `relevant_paths` entries are relative to (e.g.
        the parent of the knowledge-base directory, so a gold entry like
        "knowledge_base/architecture/x.md" resolves to a real file).
    relevant_paths : Iterable[str]
        Every `GoldExample.relevant_documents` entry across a gold file;
        duplicates are fine, each unique path is only processed once.
    config : AppConfig
        Used to build the same loader/cleaner/chunker chain
        `IngestionPipeline` uses, so results reflect real ingestion
        behavior instead of a re-implementation of it.

    Returns
    -------
    dict[str, set[str]]
        Maps each resolvable relative path to the set of `content_type`
        values its chunks were tagged with (e.g. `{"prose", "table"}`).
        Paths that don't resolve to a file on disk are skipped silently --
        `scripts/validate_gold_file.py` is responsible for flagging that.
    """
    cleaner = DefaultCleaner()
    chunker = get_chunker(config.chunking)
    result: dict[str, set[str]] = {}
    for relative_path in set(relevant_paths):
        file_path = data_root / relative_path
        if not file_path.is_file():
            continue
        raw = get_loader(file_path).load(file_path)
        cleaned = cleaner.clean(raw.content)
        spans = chunker.split(cleaned, source_type=raw.source_type)
        result[relative_path] = {span.content_type or "prose" for span in spans}
    return result


def classify_example(example: GoldExample, document_content_types: Mapping[str, set[str]]) -> str:
    """Bucket one gold example for the content-type breakdown report.

    Priority order: `unanswerable` (schema field) > `multi_hop`
    (`question_type == "multi_hop"`) > the content type(s) of its
    `relevant_documents`, else `"prose"`.

    When a referenced document contains *both* a table and a chart (two
    real TechFusion documents do), a keyword check on the question text
    (`chart`/`caption`/`trend`) prefers the `chart` bucket; otherwise
    `_BUCKET_PRIORITY` breaks the tie deterministically. This is a
    documented judgment call, not a semantic guarantee. A table-focused
    question about one of those two documents could still land in the
    wrong bucket.

    Parameters
    ----------
    example : GoldExample
        The gold question to classify.
    document_content_types : Mapping[str, set[str]]
        Output of `build_document_content_types`.

    Returns
    -------
    str
        One of `"unanswerable"`, `"multi_hop"`, `"table"`,
        `"code_configuration"`, `"chart"`, `"prose"`.
    """
    if example.unanswerable:
        return "unanswerable"
    if example.question_type == "multi_hop":
        return "multi_hop"

    buckets: set[str] = set()
    for relative_path in example.relevant_documents:
        for content_type in document_content_types.get(relative_path, set()):
            bucket = _CONTENT_TYPE_TO_BUCKET.get(content_type)
            if bucket:
                buckets.add(bucket)

    if not buckets:
        return "prose"
    if len(buckets) == 1:
        return next(iter(buckets))

    question_lower = example.question.lower()
    if "chart" in buckets and any(keyword in question_lower for keyword in _CHART_KEYWORDS):
        return "chart"
    for bucket in _BUCKET_PRIORITY:
        if bucket in buckets:
            return bucket
    return "prose"
