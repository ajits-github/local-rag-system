from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rag.ingestion.writer import Writer
from rag.schemas import Chunk, ChunkSpan, RawDocument
from rag.vision.base import VisionProvider


class FakeVectorStore:
    """Minimal VectorStore double -- no DB, just records what was written."""

    def __init__(self) -> None:
        """Start with no recorded chunks and an empty image-description cache."""
        self.written_chunks: list[Chunk] = []
        self._image_cache: dict[str, str] = {}

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Record `chunks` in `written_chunks` for assertions."""
        self.written_chunks.extend(chunks)

    def get_cached_image_description(self, image_checksum: str) -> str | None:
        """Return a cached description, or None on a miss."""
        return self._image_cache.get(image_checksum)

    def cache_image_description(
        self,
        image_checksum: str,
        source_path: str,
        provider: str,
        model_name: str,
        description: str,
    ) -> None:
        """Record a description in the fake in-memory cache."""
        self._image_cache[image_checksum] = description


class FakeVisionProvider(VisionProvider):
    """Minimal VisionProvider double: returns a canned description, records calls."""

    provider_name = "fake"
    model_name = "fake-model"

    def __init__(self, description: str = "A diagram showing X connected to Y.") -> None:
        """Store the canned description and start with no recorded calls."""
        self._description = description
        self.calls: list[tuple[Path, str | None]] = []

    def describe_image(self, image_path: Path, alt_text: str | None = None) -> str:
        """Record the call and return the canned description -- never a real network call."""
        self.calls.append((image_path, alt_text))
        return self._description


class FakeEmbedder:
    """Minimal Embedder double: returns a fixed placeholder vector per text."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one placeholder vector per input text."""
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return a placeholder vector."""
        return [0.0]


def _raw_document() -> RawDocument:
    """Build a minimal RawDocument for writer tests."""
    now = datetime.now(UTC)
    return RawDocument(
        content="irrelevant",
        source="doc.md",
        source_type="markdown",
        created_at=now,
        last_modified=now,
    )


def test_span_with_no_content_type_defaults_to_prose():
    """A ChunkSpan that leaves content_type unset is persisted as 'prose', never None."""
    vectorstore = FakeVectorStore()
    writer = Writer(FakeEmbedder(), vectorstore)

    chunks = writer.write(
        _raw_document(), "doc-1", [ChunkSpan(text="plain prose")], dataset_id="ds"
    )

    assert chunks[0].metadata.content_type == "prose"


def test_per_span_metadata_varies_chunk_to_chunk():
    """Each chunk's metadata reflects its own span, not the first/last span in the document."""
    vectorstore = FakeVectorStore()
    writer = Writer(FakeEmbedder(), vectorstore)
    spans = [
        ChunkSpan(text="a table row", content_type="table", table_headers=["Name", "Value"]),
        ChunkSpan(text="def f(): pass", content_type="code", code_language="python"),
        ChunkSpan(text="plain prose paragraph"),
    ]

    chunks = writer.write(_raw_document(), "doc-1", spans, dataset_id="ds")

    assert [c.metadata.content_type for c in chunks] == ["table", "code", "prose"]
    assert chunks[0].metadata.table_headers == ["Name", "Value"]
    assert chunks[1].metadata.code_language == "python"
    assert chunks[2].metadata.table_headers is None
    assert chunks[2].metadata.code_language is None


def test_write_persists_chunks_to_vectorstore():
    """write() calls add_chunks with every constructed Chunk."""
    vectorstore = FakeVectorStore()
    writer = Writer(FakeEmbedder(), vectorstore)
    spans = [ChunkSpan(text="one"), ChunkSpan(text="two")]

    chunks = writer.write(_raw_document(), "doc-1", spans, dataset_id="ds")

    assert vectorstore.written_chunks == chunks
    assert [c.id for c in chunks] == ["doc-1_0", "doc-1_1"]


def test_span_matching_sensitive_pattern_is_tagged_with_field_id():
    """A span containing the synthetic admin-key pattern is tagged at write time.

    Field-level-safety milestone: this ingestion-time tag (see
    rag.retrieval.field_policy.detect_sensitive_field_ids) is what lets
    RetrievalPipeline skip the redaction pass entirely for untagged
    chunks -- it carries no role decision by itself.
    """
    vectorstore = FakeVectorStore()
    writer = Writer(FakeEmbedder(), vectorstore)
    spans = [ChunkSpan(text="The synthetic test key is SYNTHETIC_ONLY_ALPHA_KEY_7Q4M_DO_NOT_USE.")]

    chunks = writer.write(_raw_document(), "doc-1", spans, dataset_id="ds")

    assert chunks[0].metadata.sensitive_field_ids == ["synthetic_admin_credential"]


def test_span_with_no_sensitive_pattern_has_no_tag():
    """An ordinary span is left with sensitive_field_ids=None, not an empty list."""
    vectorstore = FakeVectorStore()
    writer = Writer(FakeEmbedder(), vectorstore)
    spans = [ChunkSpan(text="The retry delay is 45 seconds.")]

    chunks = writer.write(_raw_document(), "doc-1", spans, dataset_id="ds")

    assert chunks[0].metadata.sensitive_field_ids is None


def test_non_prose_span_parent_is_nearest_preceding_prose_in_same_section():
    """A table/code/image span links to the last prose chunk seen in its section_path."""
    writer = Writer(FakeEmbedder(), FakeVectorStore())
    spans = [
        ChunkSpan(text="intro paragraph", section_path="Intro"),
        ChunkSpan(text="| a |\n|---|\n| 1 |", content_type="table", section_path="Intro"),
        ChunkSpan(text="![x](images/x.png)", content_type="image", section_path="Intro"),
    ]

    chunks = writer.write(_raw_document(), "doc-1", spans, dataset_id="ds")

    assert chunks[0].metadata.parent_chunk_id is None
    assert chunks[1].metadata.parent_chunk_id == chunks[0].id
    assert chunks[2].metadata.parent_chunk_id == chunks[0].id


def test_non_prose_span_with_no_preceding_prose_has_no_parent():
    """A table opening a section before any prose gets parent_chunk_id=None."""
    writer = Writer(FakeEmbedder(), FakeVectorStore())
    spans = [ChunkSpan(text="| a |\n|---|\n| 1 |", content_type="table", section_path="Setup")]

    chunks = writer.write(_raw_document(), "doc-1", spans, dataset_id="ds")

    assert chunks[0].metadata.parent_chunk_id is None


def test_parent_tracking_resets_across_sections():
    """A non-prose span never inherits a parent from a different section_path."""
    writer = Writer(FakeEmbedder(), FakeVectorStore())
    spans = [
        ChunkSpan(text="section A intro", section_path="A"),
        ChunkSpan(text="section B intro", section_path="B"),
        ChunkSpan(text="code in A", content_type="code", section_path="A"),
    ]

    chunks = writer.write(_raw_document(), "doc-1", spans, dataset_id="ds")

    # The code span belongs to section "A", so its parent is the "A" intro
    # chunk (chunks[0]), not the more recent "B" intro chunk (chunks[1]).
    assert chunks[2].metadata.parent_chunk_id == chunks[0].id


def test_no_vision_provider_leaves_image_span_untouched():
    """Without a VisionProvider, an image span is persisted as-is -- no sibling, no file access."""
    writer = Writer(FakeEmbedder(), FakeVectorStore())
    spans = [
        ChunkSpan(
            text="![x](images/missing.png)",
            content_type="image",
            attachment_name="missing.png",
            source_anchor="images/missing.png",
        )
    ]

    chunks = writer.write(_raw_document(), "doc-1", spans, dataset_id="ds")

    assert len(chunks) == 1
    assert chunks[0].metadata.vision_generated is False


def test_vision_provider_adds_sibling_chunk_without_modifying_caption_chunk(tmp_path):
    """A configured VisionProvider adds a second, vision_generated=True chunk per image span.

    The original caption/alt-text chunk is never modified -- "never
    overwrite captions with generated descriptions."
    """
    doc_dir = tmp_path / "knowledge_base"
    (doc_dir / "images").mkdir(parents=True)
    image_path = doc_dir / "images" / "diagram.png"
    image_path.write_bytes(b"fake-png-bytes")

    document = RawDocument(
        content="irrelevant",
        source=str(doc_dir / "doc.md"),
        source_type="markdown",
        created_at=datetime.now(UTC),
        last_modified=datetime.now(UTC),
    )
    vectorstore = FakeVectorStore()
    vision_provider = FakeVisionProvider("A box labeled X connects to a box labeled Y.")
    writer = Writer(FakeEmbedder(), vectorstore, vision_provider)
    spans = [
        ChunkSpan(
            text="![Architecture diagram](images/diagram.png)\n\n*Figure 1: Caption text.*",
            content_type="image",
            attachment_name="diagram.png",
            source_anchor="images/diagram.png",
            section_path="Architecture",
        )
    ]

    chunks = writer.write(document, "doc-1", spans, dataset_id="ds")

    assert len(chunks) == 2
    caption_chunk, vision_chunk = chunks
    assert caption_chunk.metadata.vision_generated is False
    assert "Figure 1: Caption text." in caption_chunk.content
    description = "A box labeled X connects to a box labeled Y."
    assert vision_chunk.metadata.vision_generated is True
    assert vision_chunk.metadata.vision_description == description
    assert vision_chunk.content == description
    assert vision_chunk.metadata.attachment_name == "diagram.png"
    assert vision_chunk.metadata.source_anchor == "images/diagram.png"
    assert vision_chunk.metadata.section_path == "Architecture"
    assert len(vision_provider.calls) == 1


def test_vision_provider_is_not_called_again_on_cache_hit(tmp_path):
    """A second write() over the same image reuses the cache -- no second describe_image call."""
    doc_dir = tmp_path / "knowledge_base"
    (doc_dir / "images").mkdir(parents=True)
    (doc_dir / "images" / "diagram.png").write_bytes(b"fake-png-bytes")

    document = RawDocument(
        content="irrelevant",
        source=str(doc_dir / "doc.md"),
        source_type="markdown",
        created_at=datetime.now(UTC),
        last_modified=datetime.now(UTC),
    )
    vectorstore = FakeVectorStore()
    vision_provider = FakeVisionProvider()
    span = ChunkSpan(
        text="![Architecture diagram](images/diagram.png)",
        content_type="image",
        attachment_name="diagram.png",
        source_anchor="images/diagram.png",
    )
    writer = Writer(FakeEmbedder(), vectorstore, vision_provider)

    writer.write(document, "doc-1", [span], dataset_id="ds")
    writer.write(document, "doc-2", [span], dataset_id="ds")

    assert len(vision_provider.calls) == 1


def test_vision_provider_skips_image_with_missing_asset_file(tmp_path):
    """An image span whose asset file isn't found on disk gets no vision sibling."""
    doc_dir = tmp_path / "knowledge_base"
    doc_dir.mkdir()
    document = RawDocument(
        content="irrelevant",
        source=str(doc_dir / "doc.md"),
        source_type="markdown",
        created_at=datetime.now(UTC),
        last_modified=datetime.now(UTC),
    )
    vectorstore = FakeVectorStore()
    vision_provider = FakeVisionProvider()
    writer = Writer(FakeEmbedder(), vectorstore, vision_provider)
    spans = [
        ChunkSpan(
            text="![x](images/missing.png)",
            content_type="image",
            attachment_name="missing.png",
            source_anchor="images/missing.png",
        )
    ]

    chunks = writer.write(document, "doc-1", spans, dataset_id="ds")

    assert len(chunks) == 1
    assert vision_provider.calls == []
