"""Corpus/gold lineage snapshot: what exactly was scored, for one eval run.

Before recording an experiment, capture `dataset_id`/`corpus_version`/
document count/chunk count/image count/canonical gold-record count/
gold-file digest/a deterministic corpus digest, so two experiments
claiming to run against the same dataset can be checked for whether they
actually scored the same corpus and gold file. Read-only against
`VectorStore`; never mutates anything. `corpus_version` is a free-form
string the caller supplies. No version-numbering scheme is invented
here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from rag.vectorstore.base import VectorStore


def _corpus_digest(checksums: dict[str, str]) -> str:
    """sha256 over sorted "{source}:{checksum}" lines. A corpus-content fingerprint.

    Deterministic regardless of ingestion order (sorted by source) and
    independent of chunking parameters (built from `documents.checksum`,
    the raw file's sha256, not chunk content). Two experiments with the
    same `corpus_digest` ingested byte-identical source files, even if
    chunking/embedding differed between them.
    """
    lines = sorted(f"{source}:{checksum}" for source, checksum in checksums.items())
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def compute_corpus_lineage(
    vectorstore: VectorStore,
    dataset_id: str,
    corpus_version: str,
    gold_path: Path,
) -> dict[str, Any]:
    """Snapshot corpus/gold identity for `dataset_id`, as of right now.

    Parameters
    ----------
    vectorstore : VectorStore
        Vector store to read corpus statistics from.
    dataset_id : str
        Namespace to snapshot.
    corpus_version : str
        Caller-supplied free-form version label; not auto-generated.
    gold_path : Path
        Path to the gold JSONL file this eval run scores against.

    Returns
    -------
    dict[str, Any]
        ``{"dataset_id", "corpus_version", "document_count", "chunk_count",
        "image_count", "active_document_count", "superseded_document_count",
        "tenant_count", "gold_record_count", "gold_file_sha256",
        "corpus_digest"}``. `active_document_count`/`superseded_document_count`
        only count documents with a `status` of exactly `"active"`/
        `"superseded"`. Documents with no `status` at all (untenanted
        legacy content) are counted in neither. `tenant_count` counts
        distinct non-null `tenant_id` values.
    """
    checksums = vectorstore.get_document_checksums(dataset_id)
    chunk_counts_by_document = vectorstore.count_chunks_by_document(dataset_id)
    content_type_counts = vectorstore.count_chunks_by_content_type(dataset_id)
    versions = vectorstore.list_document_versions(dataset_id)

    gold_bytes = gold_path.read_bytes()
    gold_record_count = sum(1 for line in gold_bytes.splitlines() if line.strip())

    return {
        "dataset_id": dataset_id,
        "corpus_version": corpus_version,
        "document_count": len(checksums),
        "chunk_count": sum(chunk_counts_by_document.values()),
        "image_count": content_type_counts.get("image", 0),
        "active_document_count": sum(1 for v in versions if v.status == "active"),
        "superseded_document_count": sum(1 for v in versions if v.status == "superseded"),
        "tenant_count": len({v.tenant_id for v in versions if v.tenant_id}),
        "gold_record_count": gold_record_count,
        "gold_file_sha256": hashlib.sha256(gold_bytes).hexdigest(),
        "corpus_digest": _corpus_digest(checksums),
    }
