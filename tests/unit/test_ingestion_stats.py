from __future__ import annotations

from pathlib import Path

from rag.config import load_config
from rag.ingestion.pipeline import IngestionPipeline, _source_is_under_root
from rag.schemas import Chunk


class StatefulFakeVectorStore:
    """Stateful VectorStore double tracking documents across repeated ingest_path calls.

    Unlike test_ingestion_category.py's FakeVectorStore (every file is
    always "new"), this one remembers checksums/chunk counts across calls,
    so it can exercise real new/changed/unchanged/deleted classification
    and the chunks_embedded/chunks_reused aggregate counts.
    """

    def __init__(self) -> None:
        """Start with no persisted documents."""
        self._documents: dict[tuple[str, str], tuple[str, str]] = {}  # (source, ds) -> (id, sum)
        self._chunk_counts: dict[str, int] = {}
        self.written_chunks: list[Chunk] = []
        self._next_id = 0

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True

    def get_or_create_document_id(self, source: str, checksum: str, dataset_id: str):
        """Assign a stable id per (source, dataset_id), reporting whether the checksum changed."""
        key = (source, dataset_id)
        if key not in self._documents:
            self._next_id += 1
            doc_id = f"doc-{self._next_id}"
            self._documents[key] = (doc_id, checksum)
            return doc_id, True
        doc_id, existing_checksum = self._documents[key]
        changed = existing_checksum != checksum
        if changed:
            self._documents[key] = (doc_id, checksum)
        return doc_id, changed

    def delete_chunks_by_document_id(self, document_id: str) -> None:
        """Remove every written chunk for `document_id` and reset its chunk count."""
        self.written_chunks = [
            c for c in self.written_chunks if c.metadata.document_id != document_id
        ]
        self._chunk_counts.pop(document_id, None)

    def delete_document(self, document_id: str) -> None:
        """Remove a document's chunks (test double has no separate documents row)."""
        self.delete_chunks_by_document_id(document_id)

    def delete_dataset(self, dataset_id: str) -> None:
        """Remove every document tracked under `dataset_id`."""
        self._documents = {k: v for k, v in self._documents.items() if k[1] != dataset_id}

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Record `chunks` and tally per-document chunk counts."""
        self.written_chunks.extend(chunks)
        for c in chunks:
            self._chunk_counts[c.metadata.document_id] = (
                self._chunk_counts.get(c.metadata.document_id, 0) + 1
            )

    def search(self, *args, **kwargs):
        """Return no results, always; unused by these tests."""
        return []

    def list_document_sources(self, dataset_id: str) -> list[str]:
        """List every source currently tracked under `dataset_id`."""
        return [source for source, ds in self._documents if ds == dataset_id]

    def delete_documents_by_source(self, dataset_id: str, sources: list[str]) -> int:
        """Delete documents by exact source match within `dataset_id`."""
        deleted = 0
        for source in sources:
            key = (source, dataset_id)
            if key in self._documents:
                doc_id, _ = self._documents.pop(key)
                self.delete_chunks_by_document_id(doc_id)
                deleted += 1
        return deleted

    def count_chunks_by_document(self, dataset_id: str) -> dict[str, int]:
        """Return `{document_id: chunk_count}` for every document in `dataset_id`."""
        ids = {
            doc_id for (_source, ds), (doc_id, _sum) in self._documents.items() if ds == dataset_id
        }
        return {doc_id: count for doc_id, count in self._chunk_counts.items() if doc_id in ids}


class FakeEmbedder:
    """Minimal Embedder double returning a fixed placeholder vector."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one placeholder vector per input text."""
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return a placeholder vector."""
        return [0.0]


def _pipeline() -> tuple[IngestionPipeline, StatefulFakeVectorStore]:
    """Build an IngestionPipeline wired to the stateful fake vectorstore/embedder."""
    vectorstore = StatefulFakeVectorStore()
    pipeline = IngestionPipeline(load_config(), vectorstore=vectorstore, embedder=FakeEmbedder())
    return pipeline, vectorstore


def test_first_ingestion_reports_every_file_as_new(tmp_path: Path):
    """A first-time directory ingest classifies every discovered file as 'new'."""
    root = tmp_path / "kb"
    root.mkdir()
    (root / "a.md").write_text("Alpha content here.", encoding="utf-8")
    (root / "b.md").write_text("Beta content here.", encoding="utf-8")

    pipeline, _vs = _pipeline()
    stats = pipeline.ingest_path(root, "test-dataset")

    assert stats.discovered == 2
    assert stats.new == 2
    assert stats.changed == 0
    assert stats.unchanged == 0
    assert stats.deleted == 0
    assert stats.chunks_embedded > 0
    assert stats.chunks_reused == 0
    assert {r["status"] for r in stats.results} == {"new"}


def test_reingesting_unchanged_files_reports_unchanged_and_reused_chunks(tmp_path: Path):
    """A second identical ingest reports every file 'unchanged' and sums their reused chunks."""
    root = tmp_path / "kb"
    root.mkdir()
    (root / "a.md").write_text("Alpha content here.", encoding="utf-8")

    pipeline, _vs = _pipeline()
    first = pipeline.ingest_path(root, "test-dataset")
    second = pipeline.ingest_path(root, "test-dataset")

    assert second.new == 0
    assert second.changed == 0
    assert second.unchanged == 1
    assert second.chunks_embedded == 0
    assert second.chunks_reused == first.chunks_embedded  # exactly what was written the first time


def test_modified_file_is_reported_as_changed_and_rewrites_chunks(tmp_path: Path):
    """Editing a file between ingests reports it as 'changed', re-embedding its chunks."""
    root = tmp_path / "kb"
    root.mkdir()
    path = root / "a.md"
    path.write_text("Alpha content here.", encoding="utf-8")

    pipeline, vs = _pipeline()
    pipeline.ingest_path(root, "test-dataset")
    path.write_text("Alpha content, now revised and longer.", encoding="utf-8")
    stats = pipeline.ingest_path(root, "test-dataset")

    assert stats.new == 0
    assert stats.changed == 1
    assert stats.unchanged == 0
    assert stats.chunks_embedded > 0
    assert any("revised" in c.content for c in vs.written_chunks)


def test_deleted_file_is_detected_and_removed(tmp_path: Path):
    """A file removed from disk between ingests is detected and its document deleted."""
    root = tmp_path / "kb"
    root.mkdir()
    keep_path = root / "keep.md"
    remove_path = root / "remove.md"
    keep_path.write_text("Keep this content.", encoding="utf-8")
    remove_path.write_text("Remove this content.", encoding="utf-8")

    pipeline, vs = _pipeline()
    pipeline.ingest_path(root, "test-dataset")
    remove_path.unlink()
    stats = pipeline.ingest_path(root, "test-dataset")

    assert stats.discovered == 1  # only keep.md discovered this run
    assert stats.deleted == 1
    assert all("Remove this" not in c.content for c in vs.written_chunks)
    assert vs.list_document_sources("test-dataset") == [str(keep_path)]


def test_single_file_target_never_deletes_other_documents(tmp_path: Path):
    """Targeting a single file (not a directory) never triggers deleted-document detection."""
    root = tmp_path / "kb"
    root.mkdir()
    a_path = root / "a.md"
    b_path = root / "b.md"
    a_path.write_text("Alpha.", encoding="utf-8")
    b_path.write_text("Beta.", encoding="utf-8")

    pipeline, vs = _pipeline()
    pipeline.ingest_path(root, "test-dataset")
    stats = pipeline.ingest_path(a_path, "test-dataset")

    assert stats.deleted == 0
    assert len(vs.list_document_sources("test-dataset")) == 2  # b.md untouched


def test_upload_style_document_outside_the_ingestion_root_survives_a_directory_ingest(
    tmp_path: Path,
):
    """A document ingested via ingest_file (e.g. POST /ingest) is never treated as deleted.

    Regression test for A1: deletion detection used to diff the full set of
    a dataset's known sources against only what this directory walk
    discovered, so a document from outside the walked root (an upload, or
    a different root) was wrongly treated as deleted.
    """
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    uploaded = uploads_dir / "c.md"
    uploaded.write_text("Uploaded content.", encoding="utf-8")

    kb_root = tmp_path / "kb"
    kb_root.mkdir()
    (kb_root / "a.md").write_text("Alpha content.", encoding="utf-8")

    pipeline, vs = _pipeline()
    pipeline.ingest_file(uploaded, "test-dataset")
    stats = pipeline.ingest_path(kb_root, "test-dataset")

    assert stats.deleted == 0
    assert str(uploaded) in vs.list_document_sources("test-dataset")


def test_documents_from_a_different_ingestion_root_survive_a_directory_ingest(tmp_path: Path):
    """Ingesting root_B never deletes documents previously ingested from root_A."""
    root_a = tmp_path / "kb_a"
    root_a.mkdir()
    (root_a / "a.md").write_text("Root A content.", encoding="utf-8")

    root_b = tmp_path / "kb_b"
    root_b.mkdir()
    (root_b / "b.md").write_text("Root B content.", encoding="utf-8")

    pipeline, vs = _pipeline()
    pipeline.ingest_path(root_a, "test-dataset")
    stats = pipeline.ingest_path(root_b, "test-dataset")

    assert stats.deleted == 0
    sources = vs.list_document_sources("test-dataset")
    assert str(root_a / "a.md") in sources
    assert str(root_b / "b.md") in sources


def test_deleted_nested_file_is_still_detected_and_removed(tmp_path: Path):
    """Deletion detection covers nested subdirectories, not just top-level files, for a root."""
    root = tmp_path / "kb"
    (root / "sub" / "deep").mkdir(parents=True)
    keep_path = root / "keep.md"
    remove_path = root / "sub" / "deep" / "remove.md"
    keep_path.write_text("Keep this content.", encoding="utf-8")
    remove_path.write_text("Remove this content.", encoding="utf-8")

    pipeline, vs = _pipeline()
    pipeline.ingest_path(root, "test-dataset")
    remove_path.unlink()
    stats = pipeline.ingest_path(root, "test-dataset")

    assert stats.discovered == 1  # only keep.md discovered this run
    assert stats.deleted == 1
    assert vs.list_document_sources("test-dataset") == [str(keep_path)]


def test_repeated_ingestion_of_the_same_root_causes_no_false_deletions(tmp_path: Path):
    """Ingesting the same, unchanged root three times in a row never reports a deletion."""
    root = tmp_path / "kb"
    root.mkdir()
    (root / "a.md").write_text("Alpha content.", encoding="utf-8")
    (root / "b.md").write_text("Beta content.", encoding="utf-8")

    pipeline, _vs = _pipeline()
    pipeline.ingest_path(root, "test-dataset")
    pipeline.ingest_path(root, "test-dataset")
    stats = pipeline.ingest_path(root, "test-dataset")

    assert stats.deleted == 0
    assert stats.unchanged == 2


def test_deletion_detection_stays_scoped_to_its_own_dataset(tmp_path: Path):
    """Re-ingesting root_A for dataset X never deletes dataset Y's documents from the same root."""
    root = tmp_path / "kb"
    root.mkdir()
    keep_path = root / "keep.md"
    remove_path = root / "remove.md"
    keep_path.write_text("Keep this content.", encoding="utf-8")
    remove_path.write_text("Remove this content.", encoding="utf-8")

    pipeline, vs = _pipeline()
    pipeline.ingest_path(root, "dataset-x")
    pipeline.ingest_path(root, "dataset-y")
    remove_path.unlink()
    stats = pipeline.ingest_path(root, "dataset-x")

    assert stats.deleted == 1
    assert vs.list_document_sources("dataset-x") == [str(keep_path)]
    assert sorted(vs.list_document_sources("dataset-y")) == sorted(
        [str(keep_path), str(remove_path)]
    )


def test_same_root_reingested_with_a_different_spelling_replaces_without_duplicating(
    tmp_path: Path, monkeypatch
):
    """Re-spelling the same root between runs churns identity but never duplicates or loses content.

    Known, bounded limitation adjacent to A1, not a reintroduction of it.
    `document_id` is keyed on the literal `source` string everywhere in
    this system (see the project's "Document identity" docs), not just in
    deletion detection -- a differently-spelled root produces a
    differently-spelled `source` for the same physical file, which this
    system already treats as a distinct identity. `_source_is_under_root`'s
    scoping guarantees no *cross-root* document is ever wrongly deleted; it
    does not, and given the existing string-keyed identity model safely
    cannot, guarantee document_id continuity when the same root is spelled
    differently across runs. Normalizing every discovered source to a
    resolved/absolute form would avoid this specific churn but would also
    make every already-ingested (relative-form) document in every existing
    dataset mismatch on its very next re-ingestion, forcing a one-time mass
    delete-and-recreate well out of proportion to this edge case, so it was
    deliberately not done. What's verified here is the safe fallback: the
    old spelling's document is replaced, never left as an orphaned
    duplicate alongside the new one, and its content is never lost.
    """
    root = tmp_path / "kb"
    root.mkdir()
    (root / "a.md").write_text("Alpha content.", encoding="utf-8")

    pipeline, vs = _pipeline()
    monkeypatch.chdir(tmp_path)
    pipeline.ingest_path(Path("kb"), "test-dataset")
    chunks_after_first_run = len(vs.written_chunks)

    absolute_root = (tmp_path / "kb").resolve()
    stats = pipeline.ingest_path(absolute_root, "test-dataset")

    assert stats.deleted == 1
    assert stats.new == 1
    assert len(vs.list_document_sources("test-dataset")) == 1  # never duplicated
    assert len(vs.written_chunks) == chunks_after_first_run  # content survives, not lost


def test_posix_style_persisted_source_vs_native_discovered_path_never_duplicates(
    tmp_path: Path, monkeypatch
):
    """A source persisted with POSIX separators (e.g. a Linux container run) is safely replaced.

    Simulates the cross-platform half of the same limitation described in
    test_same_root_reingested_with_a_different_spelling_replaces_without_duplicating:
    a document ingested elsewhere with forward-slash separators, rediscovered
    on this host with native separators. The safe invariant holds:
    replaced, not duplicated.
    """
    root = tmp_path / "kb"
    root.mkdir()
    (root / "a.md").write_text("Alpha content.", encoding="utf-8")

    pipeline, vs = _pipeline()
    monkeypatch.chdir(tmp_path)

    posix_source = str(root / "a.md").replace("\\", "/")
    vs._documents[(posix_source, "test-dataset")] = ("doc-preexisting", "stale-checksum")

    stats = pipeline.ingest_path(Path("kb"), "test-dataset")

    assert stats.deleted == 1  # the differently-separated entry is replaced, not left orphaned
    assert len(vs.list_document_sources("test-dataset")) == 1  # never duplicated


class TestSourceIsUnderRoot:
    """Direct coverage of the resolved-path root-membership helper the A1 fix relies on."""

    def test_direct_child_is_under_root(self, tmp_path: Path):
        """A file directly inside the root is in scope."""
        root = tmp_path / "kb"
        assert _source_is_under_root(str(root / "a.md"), root) is True

    def test_nested_descendant_is_under_root(self, tmp_path: Path):
        """A file several subdirectories deep is still in scope."""
        root = tmp_path / "kb"
        assert _source_is_under_root(str(root / "sub" / "deep" / "a.md"), root) is True

    def test_sibling_directory_sharing_a_name_prefix_is_not_under_root(self, tmp_path: Path):
        """A sibling directory whose name starts with the root's name is not a false match.

        Guards against naive raw string prefix matching, which would wrongly
        treat 'kb2' as being under 'kb'.
        """
        root = tmp_path / "kb"
        sibling_source = str(tmp_path / "kb2" / "a.md")
        assert _source_is_under_root(sibling_source, root) is False

    def test_unrelated_root_is_not_under_root(self, tmp_path: Path):
        """A document from an unrelated directory (e.g. an upload folder) is out of scope."""
        root = tmp_path / "kb"
        other_source = str(tmp_path / "uploads" / "a.md")
        assert _source_is_under_root(other_source, root) is False

    def test_relative_and_absolute_forms_of_the_same_file_agree(self, tmp_path: Path, monkeypatch):
        """A relative source and its absolute equivalent both resolve to the same membership."""
        monkeypatch.chdir(tmp_path)
        relative_source = "kb/a.md"
        absolute_source = str(tmp_path / "kb" / "a.md")
        assert _source_is_under_root(relative_source, Path("kb")) is True
        assert _source_is_under_root(absolute_source, Path("kb")) is True

    def test_dot_prefixed_root_matches_its_plain_relative_equivalent(
        self, tmp_path: Path, monkeypatch
    ):
        """`./kb` and `kb` as the root resolve to the same scope for the same source."""
        monkeypatch.chdir(tmp_path)
        source = "kb/a.md"
        assert _source_is_under_root(source, Path("./kb")) is True

    def test_windows_style_separators_normalize_against_a_posix_root(
        self, tmp_path: Path, monkeypatch
    ):
        """A source with backslashes still compares correctly against a forward-slash root."""
        monkeypatch.chdir(tmp_path)
        windows_style_source = "kb\\sub\\a.md"
        assert _source_is_under_root(windows_style_source, Path("kb")) is True
