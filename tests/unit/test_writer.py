from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rag.ingestion.writer import Writer
from rag.schemas import Chunk, ChunkSpan, RawDocument
from rag.vision.base import VisionProvider


class FakeVectorStore:
    """Minimal VectorStore double. No DB, just records what was written."""

    def __init__(self) -> None:
        """Start with no recorded chunks and an empty image-description cache."""
        self.written_chunks: list[Chunk] = []
        self._image_cache: dict[str, str] = {}

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Record `chunks` in `written_chunks` for assertions."""
        self.written_chunks.extend(chunks)

    def get_cached_image_description(
        self, image_checksum: str, provider: str, model_name: str, prompt_version: str
    ) -> str | None:
        """Return a cached description, or None on a miss."""
        return self._image_cache.get((image_checksum, provider, model_name, prompt_version))

    def cache_image_description(
        self,
        image_checksum: str,
        source_path: str,
        provider: str,
        model_name: str,
        prompt_version: str,
        description: str,
    ) -> None:
        """Record a description in the fake in-memory cache."""
        self._image_cache[(image_checksum, provider, model_name, prompt_version)] = description


class FakeVisionProvider(VisionProvider):
    """Minimal VisionProvider double: returns a canned description, records calls."""

    provider_name = "fake"
    model_name = "fake-model"
    prompt_version = "v1"

    def __init__(
        self,
        description: str = "A diagram showing X connected to Y.",
        prompt_version: str = "v1",
        provider_name: str = "fake",
        model_name: str = "fake-model",
        raises: Exception | None = None,
    ) -> None:
        """Store the canned description and start with no recorded calls.

        `raises`, when set, is raised by `describe_image` instead of
        returning `description`. It simulates a provider call failure
        (Ollama unreachable/timed out/malformed response) without a real
        network call.
        """
        self._description = description
        self.prompt_version = prompt_version
        self.provider_name = provider_name
        self.model_name = model_name
        self._raises = raises
        self.calls: list[tuple[Path, str | None]] = []

    def describe_image(self, image_path: Path, alt_text: str | None = None) -> str:
        """Record the call and return the canned description without a network call."""
        self.calls.append((image_path, alt_text))
        if self._raises is not None:
            raise self._raises
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

    This ingestion-time tag lets RetrievalPipeline skip content redaction
    for untagged chunks. Metadata redaction still runs independently. The
    tag carries no role decision by itself.
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


def test_span_with_sensitive_pattern_only_in_section_path_is_still_tagged():
    """A credential embedded only in a heading is tagged, not just body text.

    Ingestion-time detection covers scannable metadata because downstream
    duplicate diagnostics and egress checks read `sensitive_field_ids`
    directly.
    """
    vectorstore = FakeVectorStore()
    writer = Writer(FakeEmbedder(), vectorstore)
    spans = [
        ChunkSpan(
            text="See the admin setup steps below.",
            section_path="Admin Credentials > SYNTHETIC_ONLY_ALPHA_KEY_7Q4M_DO_NOT_USE",
        )
    ]

    chunks = writer.write(_raw_document(), "doc-1", spans, dataset_id="ds")

    assert chunks[0].metadata.sensitive_field_ids == ["synthetic_admin_credential"]


def test_span_with_sensitive_pattern_only_in_attachment_name_is_still_tagged():
    """A credential embedded only in an attachment filename is tagged."""
    vectorstore = FakeVectorStore()
    writer = Writer(FakeEmbedder(), vectorstore)
    spans = [
        ChunkSpan(
            text="See the attached configuration export.",
            attachment_name="admin-key-SYNTHETIC_ONLY_ALPHA_KEY_7Q4M_DO_NOT_USE.pdf",
            source_anchor="assets/admin-key-SYNTHETIC_ONLY_ALPHA_KEY_7Q4M_DO_NOT_USE.pdf",
        )
    ]

    chunks = writer.write(_raw_document(), "doc-1", spans, dataset_id="ds")

    assert chunks[0].metadata.sensitive_field_ids == ["synthetic_admin_credential"]


def test_span_with_sensitive_pattern_only_in_document_source_is_still_tagged():
    """A credential embedded only in the document's own filename is tagged.

    `source` is document-level (the same for every span of a document),
    not a per-span field, so this is scanned from `document.source`
    rather than from the ChunkSpan itself. See Writer.write's call to
    combine_scannable_fields.
    """
    vectorstore = FakeVectorStore()
    writer = Writer(FakeEmbedder(), vectorstore)
    now = datetime.now(UTC)
    document = RawDocument(
        content="irrelevant",
        source="admin-key-SYNTHETIC_ONLY_ALPHA_KEY_7Q4M_DO_NOT_USE.md",
        source_type="markdown",
        created_at=now,
        last_modified=now,
    )
    spans = [ChunkSpan(text="Ordinary body text with no sensitive value at all.")]

    chunks = writer.write(document, "doc-1", spans, dataset_id="ds")

    assert chunks[0].metadata.sensitive_field_ids == ["synthetic_admin_credential"]


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
    """Without a VisionProvider, an image span is persisted as-is. No sibling, no file access."""
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

    The original caption/alt-text chunk is never overwritten by the
    generated description.
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
    """A second write() over the same image reuses the cache without another vision call."""
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


def test_vision_provider_cache_misses_on_changed_prompt_version(tmp_path):
    """Switching prompt_version (same provider/model) invalidates the cache. no stale reuse.

    Regression test for a real gap this milestone closed: the cache key
    used to be image_checksum alone, so switching provider/model/prompt
    would have silently served a description generated under different
    instructions.
    """
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
    span = ChunkSpan(
        text="![Architecture diagram](images/diagram.png)",
        content_type="image",
        attachment_name="diagram.png",
        source_anchor="images/diagram.png",
    )

    provider_v1 = FakeVisionProvider(description="v1 description", prompt_version="v1")
    Writer(FakeEmbedder(), vectorstore, provider_v1).write(
        document, "doc-1", [span], dataset_id="ds"
    )

    provider_v2 = FakeVisionProvider(description="v2 description", prompt_version="v2")
    chunks = Writer(FakeEmbedder(), vectorstore, provider_v2).write(
        document, "doc-2", [span], dataset_id="ds"
    )

    assert len(provider_v2.calls) == 1  # a real call was made, not a stale cache hit
    vision_chunk = next(c for c in chunks if c.metadata.vision_generated)
    assert vision_chunk.metadata.vision_description == "v2 description"


def test_vision_provider_cache_misses_on_changed_image_bytes(tmp_path):
    """A different image at the same path (different checksum) is never served a stale description.

    Same provider/model/prompt_version. Only the image bytes on disk
    changed between writes, so the cache key (keyed on the image's own
    sha256, not its path) must miss and re-describe.
    """
    doc_dir = tmp_path / "knowledge_base"
    (doc_dir / "images").mkdir(parents=True)
    image_path = doc_dir / "images" / "diagram.png"
    image_path.write_bytes(b"first-image-bytes")

    document = RawDocument(
        content="irrelevant",
        source=str(doc_dir / "doc.md"),
        source_type="markdown",
        created_at=datetime.now(UTC),
        last_modified=datetime.now(UTC),
    )
    vectorstore = FakeVectorStore()
    span = ChunkSpan(
        text="![Architecture diagram](images/diagram.png)",
        content_type="image",
        attachment_name="diagram.png",
        source_anchor="images/diagram.png",
    )

    provider = FakeVisionProvider(description="first description")
    Writer(FakeEmbedder(), vectorstore, provider).write(document, "doc-1", [span], dataset_id="ds")

    image_path.write_bytes(b"second-image-bytes-entirely-different")
    provider_second_call = FakeVisionProvider(description="second description")
    chunks = Writer(FakeEmbedder(), vectorstore, provider_second_call).write(
        document, "doc-2", [span], dataset_id="ds"
    )

    assert len(provider_second_call.calls) == 1  # new checksum. not a stale hit off the old bytes
    vision_chunk = next(c for c in chunks if c.metadata.vision_generated)
    assert vision_chunk.metadata.vision_description == "second description"


def test_vision_provider_cache_misses_on_changed_model_name(tmp_path):
    """Switching model_name (same provider/prompt_version) invalidates the cache.

    No stale reuse across models.
    """
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
    span = ChunkSpan(
        text="![Architecture diagram](images/diagram.png)",
        content_type="image",
        attachment_name="diagram.png",
        source_anchor="images/diagram.png",
    )

    provider_small = FakeVisionProvider(
        description="small-model description", model_name="moondream"
    )
    Writer(FakeEmbedder(), vectorstore, provider_small).write(
        document, "doc-1", [span], dataset_id="ds"
    )

    provider_large = FakeVisionProvider(
        description="large-model description", model_name="moondream-large"
    )
    chunks = Writer(FakeEmbedder(), vectorstore, provider_large).write(
        document, "doc-2", [span], dataset_id="ds"
    )

    assert len(provider_large.calls) == 1  # a real call was made, not a stale hit off the old model
    vision_chunk = next(c for c in chunks if c.metadata.vision_generated)
    assert vision_chunk.metadata.vision_description == "large-model description"


def test_vision_provider_cache_misses_on_changed_provider(tmp_path):
    """Switching provider_name (same model/prompt_version, hypothetically) invalidates the cache."""
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
    span = ChunkSpan(
        text="![Architecture diagram](images/diagram.png)",
        content_type="image",
        attachment_name="diagram.png",
        source_anchor="images/diagram.png",
    )

    provider_a = FakeVisionProvider(description="provider-a description", provider_name="ollama")
    Writer(FakeEmbedder(), vectorstore, provider_a).write(
        document, "doc-1", [span], dataset_id="ds"
    )

    provider_b = FakeVisionProvider(description="provider-b description", provider_name="hosted")
    chunks = Writer(FakeEmbedder(), vectorstore, provider_b).write(
        document, "doc-2", [span], dataset_id="ds"
    )

    assert len(provider_b.calls) == 1  # a real call was made, not a stale hit off the old provider
    vision_chunk = next(c for c in chunks if c.metadata.vision_generated)
    assert vision_chunk.metadata.vision_description == "provider-b description"


def test_vision_provider_call_failure_skips_image_without_failing_document(tmp_path):
    """A `describe_image` exception (Ollama unreachable/timeout/malformed response) is caught.

    That one image gets no vision sibling, the same outcome as a
    missing asset file. The rest of the document still ingests
    normally, and the caption/alt-text chunk for the failed image is
    still written. A single vision-model hiccup must never fail an
    entire document's ingestion.
    """
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
    vision_provider = FakeVisionProvider(raises=ConnectionError("Ollama unreachable"))
    writer = Writer(FakeEmbedder(), vectorstore, vision_provider)
    spans = [
        ChunkSpan(
            text="![Architecture diagram](images/diagram.png)",
            content_type="image",
            attachment_name="diagram.png",
            source_anchor="images/diagram.png",
        ),
        ChunkSpan(text="Unrelated prose paragraph, elsewhere in the same document."),
    ]

    chunks = writer.write(document, "doc-1", spans, dataset_id="ds")

    assert len(vision_provider.calls) == 1  # the call was attempted, and failed
    assert len(chunks) == 2  # the image caption chunk + the prose chunk. no vision sibling
    assert all(not c.metadata.vision_generated for c in chunks)
    assert chunks[1].content == "Unrelated prose paragraph, elsewhere in the same document."


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
