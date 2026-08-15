from __future__ import annotations

from pathlib import Path

from rag.eval.corpus_lineage import compute_corpus_lineage
from rag.schemas import DocumentVersionInfo


class _FakeVectorStore:
    """Minimal VectorStore double exposing only what compute_corpus_lineage calls."""

    def __init__(self, checksums, chunk_counts_by_document, content_type_counts, versions):
        self._checksums = checksums
        self._chunk_counts_by_document = chunk_counts_by_document
        self._content_type_counts = content_type_counts
        self._versions = versions

    def get_document_checksums(self, dataset_id: str):
        return self._checksums

    def count_chunks_by_document(self, dataset_id: str):
        return self._chunk_counts_by_document

    def count_chunks_by_content_type(self, dataset_id: str):
        return self._content_type_counts

    def list_document_versions(self, dataset_id: str):
        return self._versions


def _write_gold(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "gold.jsonl"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_compute_corpus_lineage_counts_and_digests(tmp_path: Path):
    """document/chunk/image counts, gold record count, and digests are all populated."""
    vectorstore = _FakeVectorStore(
        checksums={"a.md": "sum-a", "b.md": "sum-b"},
        chunk_counts_by_document={"d1": 3, "d2": 2},
        content_type_counts={"prose": 4, "image": 1},
        versions=[
            DocumentVersionInfo(document_id="d1", source="a.md", status="active"),
            DocumentVersionInfo(document_id="d2", source="b.md", status="superseded"),
        ],
    )
    gold_path = _write_gold(tmp_path, ['{"question": "q1"}', '{"question": "q2"}'])

    lineage = compute_corpus_lineage(vectorstore, "techfusion", "2026-08-14-v1", gold_path)

    assert lineage["dataset_id"] == "techfusion"
    assert lineage["corpus_version"] == "2026-08-14-v1"
    assert lineage["document_count"] == 2
    assert lineage["chunk_count"] == 5
    assert lineage["image_count"] == 1
    assert lineage["active_document_count"] == 1
    assert lineage["superseded_document_count"] == 1
    assert lineage["gold_record_count"] == 2
    assert len(lineage["gold_file_sha256"]) == 64
    assert len(lineage["corpus_digest"]) == 64


def test_compute_corpus_lineage_digest_is_order_independent(tmp_path: Path):
    """The corpus digest is identical regardless of dict insertion order (sorted internally)."""
    gold_path = _write_gold(tmp_path, ['{"question": "q1"}'])
    versions: list[DocumentVersionInfo] = []

    lineage_a = compute_corpus_lineage(
        _FakeVectorStore({"a.md": "x", "b.md": "y"}, {}, {}, versions),
        "ds",
        "v1",
        gold_path,
    )
    lineage_b = compute_corpus_lineage(
        _FakeVectorStore({"b.md": "y", "a.md": "x"}, {}, {}, versions),
        "ds",
        "v1",
        gold_path,
    )
    assert lineage_a["corpus_digest"] == lineage_b["corpus_digest"]


def test_compute_corpus_lineage_tenant_count_ignores_none(tmp_path: Path):
    """tenant_count counts distinct non-null tenant_id values only."""
    gold_path = _write_gold(tmp_path, ['{"question": "q1"}'])
    versions = [
        DocumentVersionInfo(document_id="d1", source="a.md", tenant_id="tenant_alpha"),
        DocumentVersionInfo(document_id="d2", source="b.md", tenant_id="tenant_beta"),
        DocumentVersionInfo(document_id="d3", source="c.md", tenant_id=None),
    ]
    lineage = compute_corpus_lineage(_FakeVectorStore({}, {}, {}, versions), "ds", "v1", gold_path)
    assert lineage["tenant_count"] == 2
