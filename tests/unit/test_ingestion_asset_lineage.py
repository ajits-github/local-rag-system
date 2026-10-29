"""Referenced-image lineage in ingestion (BUG-016, BUG-017).

BUG-016: the change-detection checksum covered only the document's own bytes,
so replacing or removing a referenced image under unchanged Markdown never
re-ingested it, and the stale vision chunk stayed indexed.
BUG-017: vision resolved images beside the citation path (`document.source`,
a staged upload's *final* location) instead of the file actually being read,
so a replacement upload could be described from the previous version's image.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from rag.config import load_config
from rag.ingestion.pipeline import IngestionPipeline
from rag.ingestion.writer import Writer
from rag.schemas import Chunk, ChunkSpan, RawDocument
from rag.vision.base import VisionProvider


class _VectorStore:
    """Checksum-tracking VectorStore double with an in-memory image-description cache."""

    def __init__(self) -> None:
        self._documents: dict[tuple[str, str], tuple[str, str]] = {}
        self.cached_source_paths: list[str] = []

    def get_or_create_document_id(self, source: str, checksum: str, dataset_id: str):
        key = (source, dataset_id)
        if key not in self._documents:
            self._documents[key] = (f"doc-{len(self._documents) + 1}", "")
            return self._documents[key][0], True
        doc_id, existing = self._documents[key]
        return doc_id, existing != checksum

    def replace_document_chunks(self, document_id: str, checksum: str, chunks: list[Chunk]):
        key = next(k for k, v in self._documents.items() if v[0] == document_id)
        self._documents[key] = (document_id, checksum)

    def get_cached_image_description(self, image_checksum, provider, model_name, prompt_version):
        return None

    def cache_image_description(self, *, source_path, **_):
        self.cached_source_paths.append(source_path)


class _Embedder:
    def embed_documents(self, texts):
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0]


class _Vision(VisionProvider):
    provider_name = "fake"
    model_name = "fake-model"
    prompt_version = "v1"

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def describe_image(self, image_path: Path, alt_text: str | None = None) -> str:
        self.calls.append(image_path)
        return f"description of {image_path.read_bytes().decode()}"


def _pipeline(vision: VisionProvider | None):
    store = _VectorStore()
    pipeline = IngestionPipeline(
        load_config(), vectorstore=store, embedder=_Embedder(), vision_provider=vision
    )
    return pipeline, store


def _doc_with_image(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "assets").mkdir()
    image = tmp_path / "assets" / "arch.png"
    image.write_bytes(b"version-1")
    doc = tmp_path / "doc.md"
    doc.write_text("# Arch\n\n![Architecture](assets/arch.png)\n", encoding="utf-8")
    return doc, image


def test_replacing_a_referenced_image_reingests_the_document(tmp_path: Path):
    """Same Markdown, new image bytes: the document is no longer reported unchanged."""
    doc, image = _doc_with_image(tmp_path)
    pipeline, _ = _pipeline(_Vision())
    pipeline.ingest_file(doc, "ds")

    assert pipeline.ingest_file(doc, "ds")["changed"] is False
    image.write_bytes(b"version-2")
    assert pipeline.ingest_file(doc, "ds")["changed"] is True


def test_removing_a_referenced_image_reingests_the_document(tmp_path: Path):
    """A deleted image also invalidates the document, so its vision chunk is dropped."""
    doc, image = _doc_with_image(tmp_path)
    pipeline, _ = _pipeline(_Vision())
    pipeline.ingest_file(doc, "ds")

    image.unlink()
    assert pipeline.ingest_file(doc, "ds")["changed"] is True


def test_without_vision_image_bytes_do_not_affect_the_checksum(tmp_path: Path):
    """With no vision provider, image bytes feed nothing, so they never force a re-ingest."""
    doc, image = _doc_with_image(tmp_path)
    pipeline, _ = _pipeline(None)
    pipeline.ingest_file(doc, "ds")

    image.write_bytes(b"version-2")
    assert pipeline.ingest_file(doc, "ds")["changed"] is False


def test_document_without_images_keeps_its_plain_bytes_checksum(tmp_path: Path):
    """No image references means the checksum is exactly the file's sha256, as before."""
    doc = tmp_path / "plain.md"
    doc.write_text("No images here.", encoding="utf-8")
    pipeline, _ = _pipeline(_Vision())
    raw = RawDocument(
        content=doc.read_text(encoding="utf-8"),
        source=str(doc),
        source_type="markdown",
        created_at=datetime.now(UTC),
        last_modified=datetime.now(UTC),
    )

    assert pipeline._document_checksum(doc, raw) == hashlib.sha256(doc.read_bytes()).hexdigest()


def test_vision_reads_the_staged_image_not_the_final_location(tmp_path: Path):
    """BUG-017: bytes come from `asset_base_dir`; the cache records the citation-facing path."""
    final_dir = tmp_path / "uploads"
    (final_dir / "assets").mkdir(parents=True)
    (final_dir / "assets" / "arch.png").write_bytes(b"previous-version")
    staging_dir = tmp_path / "staging"
    (staging_dir / "assets").mkdir(parents=True)
    (staging_dir / "assets" / "arch.png").write_bytes(b"new-version")

    store, vision = _VectorStore(), _Vision()
    writer = Writer(_Embedder(), store, vision)
    now = datetime.now(UTC)
    document = RawDocument(
        content="irrelevant",
        source=str(final_dir / "doc.md"),
        source_type="markdown",
        created_at=now,
        last_modified=now,
    )
    spans = [
        ChunkSpan(
            text="![Architecture](assets/arch.png)",
            content_type="image",
            source_anchor="assets/arch.png",
        )
    ]

    chunks = writer.write(document, "doc-1", spans, dataset_id="ds", asset_base_dir=staging_dir)

    assert vision.calls == [(staging_dir / "assets" / "arch.png").resolve()]
    assert any(c.content == "description of new-version" for c in chunks)
    assert store.cached_source_paths == [str(final_dir / "assets" / "arch.png")]
