"""Unit tests for `IngestionPipeline.ingest_file`'s `caller`-driven tenant governance.

Uses a minimal fake `VectorStore`/`Embedder` (no real Postgres/ML model), in
the same spirit as `test_ingestion_stats.py`'s doubles, focused specifically
on what `ChunkMetadata.tenant_id` ends up persisted as for a given
`caller`/parsed-document combination.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import load_config
from rag.ingestion.governance import IngestCallerContext, IngestGovernanceError
from rag.ingestion.pipeline import IngestionPipeline
from rag.schemas import Chunk


class _RecordingVectorStore:
    """Fake VectorStore that just records the tenant_id every written chunk carries."""

    def __init__(self) -> None:
        self.written_chunks: list[Chunk] = []
        self._next_id = 0

    def health_check(self) -> bool:
        return True

    def get_or_create_document_id(self, source: str, checksum: str, dataset_id: str):
        self._next_id += 1
        return f"doc-{self._next_id}", True

    def delete_chunks_by_document_id(self, document_id: str) -> None:
        self.written_chunks = [
            c for c in self.written_chunks if c.metadata.document_id != document_id
        ]

    def add_chunks(self, chunks: list[Chunk]) -> None:
        self.written_chunks.extend(chunks)


class _FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


def _pipeline() -> tuple[IngestionPipeline, _RecordingVectorStore]:
    vectorstore = _RecordingVectorStore()
    pipeline = IngestionPipeline(load_config(), vectorstore=vectorstore, embedder=_FakeEmbedder())
    return pipeline, vectorstore


def test_pdf_like_upload_with_no_front_matter_is_stamped_with_caller_tenant(tmp_path: Path):
    """A loader producing no governance metadata gets the caller's own tenant_id stamped.

    Stood in here by a plain `.txt` file, which, like PDF/DOCX/HTML,
    `TextLoader` never parses front matter for. [Test A / C]
    """
    path = tmp_path / "customer-incident.txt"
    path.write_text(
        "Ordinary uploaded content with no governance metadata at all.", encoding="utf-8"
    )

    pipeline, vectorstore = _pipeline()
    caller = IngestCallerContext(tenant_id="tenant_alpha", is_privileged=False)
    pipeline.ingest_file(path, "test-dataset", caller=caller)

    assert vectorstore.written_chunks, "expected at least one chunk written"
    assert all(c.metadata.tenant_id == "tenant_alpha" for c in vectorstore.written_chunks)


def test_markdown_with_matching_front_matter_tenant_is_allowed_unchanged(tmp_path: Path):
    """Explicit front-matter tenant_id equal to the caller's own tenant is preserved. [Test D]."""
    path = tmp_path / "policy.md"
    path.write_text(
        '---\ntenant_id: "tenant_alpha"\n---\n\nSame-tenant governed content.',
        encoding="utf-8",
    )

    pipeline, vectorstore = _pipeline()
    caller = IngestCallerContext(tenant_id="tenant_alpha", is_privileged=False)
    pipeline.ingest_file(path, "test-dataset", caller=caller)

    assert all(c.metadata.tenant_id == "tenant_alpha" for c in vectorstore.written_chunks)


def test_normal_caller_cannot_upload_front_matter_for_a_different_tenant(tmp_path: Path):
    """A non-privileged tenant-alpha caller uploading front matter for tenant_beta is rejected.

    Nothing is persisted. [Test E]
    """
    path = tmp_path / "cross-tenant.md"
    path.write_text(
        '---\ntenant_id: "tenant_beta"\n---\n\nAttempted cross-tenant content.',
        encoding="utf-8",
    )

    pipeline, vectorstore = _pipeline()
    caller = IngestCallerContext(tenant_id="tenant_alpha", is_privileged=False)
    with pytest.raises(IngestGovernanceError):
        pipeline.ingest_file(path, "test-dataset", caller=caller)

    assert vectorstore.written_chunks == []


def test_privileged_caller_may_upload_on_behalf_of_a_different_explicit_tenant(tmp_path: Path):
    """A caller holding a cross-tenant support role may set an explicit different tenant_id.

    Matches the existing `cross_tenant_support_roles` privilege model. [Test F]
    """
    path = tmp_path / "support-authored.md"
    path.write_text(
        '---\ntenant_id: "tenant_beta"\n---\n\nContent authored on tenant_beta\'s behalf.',
        encoding="utf-8",
    )

    pipeline, vectorstore = _pipeline()
    caller = IngestCallerContext(tenant_id="techfusion_support", is_privileged=True)
    pipeline.ingest_file(path, "test-dataset", caller=caller)

    assert all(c.metadata.tenant_id == "tenant_beta" for c in vectorstore.written_chunks)


def test_no_caller_preserves_pre_fix_behavior(tmp_path: Path):
    """`caller=None` is byte-identical to ingestion before this fix existed.

    Covers the CLI/`make ingest` and unauthenticated-API-request cases: a
    document's own (possibly `None`) tenant_id is persisted exactly as
    parsed. [Test G]
    """
    path = tmp_path / "legacy-upload.txt"
    path.write_text("No identity involved in this ingestion at all.", encoding="utf-8")

    pipeline, vectorstore = _pipeline()
    pipeline.ingest_file(path, "test-dataset")  # caller defaults to None

    assert vectorstore.written_chunks, "expected at least one chunk written"
    assert all(c.metadata.tenant_id is None for c in vectorstore.written_chunks)


def test_no_caller_preserves_explicit_front_matter_tenant_unchanged(tmp_path: Path):
    """`caller=None` never touches an already-governed document's own front-matter tenant_id.

    Proves the static/trusted corpus's existing CLI-ingested governance
    metadata (e.g. `data/knowledge_base/security_evaluation/...`) is
    completely unaffected by this fix. [Test G]
    """
    path = tmp_path / "already-governed.md"
    path.write_text(
        '---\ntenant_id: "tenant_beta"\n---\n\nAlready-governed static-corpus content.',
        encoding="utf-8",
    )

    pipeline, vectorstore = _pipeline()
    pipeline.ingest_file(path, "test-dataset")  # caller defaults to None

    assert all(c.metadata.tenant_id == "tenant_beta" for c in vectorstore.written_chunks)
