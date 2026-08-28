"""Explicit ingestion pipeline: loader -> cleaner -> chunker -> embedder -> writer.

Also runnable as a CLI: `python -m rag.ingestion.pipeline <path> --dataset-id <id>`
(used by `make ingest FILE=... DATASET_ID=...`).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Any

from rag.chunkers.registry import get_chunker
from rag.cleaners.default_cleaner import DefaultCleaner
from rag.config import AppConfig, load_config
from rag.embedders.base import Embedder
from rag.factory import build_embedder, build_vectorstore, build_vision_provider
from rag.ingestion.governance import IngestCallerContext, resolve_ingest_tenant_id
from rag.ingestion.writer import Writer
from rag.loaders.registry import get_loader
from rag.schemas import IngestionStats
from rag.vectorstore.base import VectorStore
from rag.vision.base import VisionProvider

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Runs a document through loader -> cleaner -> chunker -> embedder -> writer."""

    def __init__(
        self,
        config: AppConfig,
        vectorstore: VectorStore | None = None,
        embedder: Embedder | None = None,
        vision_provider: VisionProvider | None = None,
    ) -> None:
        """Wire up the pipeline's stages from config (or injected instances).

        Parameters
        ----------
        config : AppConfig
            Application configuration.
        vectorstore : VectorStore | None, optional
            Vector store to write to; built from `config` if omitted.
        embedder : Embedder | None, optional
            Embedder to use; built from `config` if omitted.
        vision_provider : VisionProvider | None, optional
            Vision provider passed to `Writer`; built from
            `config.vision.provider` if omitted (`None` for `"none"`, the
            only provider implemented so far. So this is `None` in
            practice until a hosted provider exists). Inject a test mock
            here to exercise vision-mode ingestion without config changes.
        """
        self._config = config
        self._vectorstore = vectorstore or build_vectorstore(config)
        self._embedder = embedder or build_embedder(config)
        self._cleaner = DefaultCleaner()
        self._chunker = get_chunker(config.chunking)
        self._writer = Writer(
            self._embedder, self._vectorstore, vision_provider or build_vision_provider(config)
        )

    def clear_dataset(self, dataset_id: str) -> None:
        """Remove every document (and cascade its chunks) tagged with dataset_id.

        For re-ingesting a namespace from a clean slate.

        Parameters
        ----------
        dataset_id : str
            The dataset namespace to clear.
        """
        self._vectorstore.delete_dataset(dataset_id)

    def ingest_file(
        self,
        path: Path,
        dataset_id: str,
        category: str | None = None,
        caller: IngestCallerContext | None = None,
        source_override: str | None = None,
    ) -> dict[str, Any]:
        """Load, clean, chunk, embed, and persist a single file.

        Skips re-writing chunks if the file's checksum is unchanged since
        the last ingestion of the same (source, dataset_id).

        Parameters
        ----------
        path : Path
            File to read content and governance front matter from. Not
            necessarily the persisted document identity; see
            `source_override`.
        dataset_id : str
            Namespace tag stored on every chunk.
        category : str | None, optional
            Folder-derived category tag, by default None.
        caller : IngestCallerContext | None, optional
            The authenticated caller's governance identity, if any. `None`
            (the default) means no caller identity is available. The CLI
            (`make ingest`)/`ingest_path` always pass this, and `POST
            /ingest` passes it too whenever JWT auth is disabled or an
            unauthenticated request was allowed through
            (`insecure_dev_mode`). In that case the loaded document's own
            `tenant_id` (front matter, or `None`) is persisted exactly as
            parsed, unchanged from before this parameter existed. When set,
            `resolve_ingest_tenant_id` resolves the effective `tenant_id`
            before any database write happens, so a rejected upload leaves
            no trace (no document row, no checksum).
        source_override : str | None, optional
            When set, overrides the loaded document's `source` (normally
            `str(path)`) before it's persisted or used to resolve
            `document_id`. `POST /ingest` reads upload bytes from a
            temporary file but must persist the stable, caller-facing
            upload path as `source`. This keeps that path's `document_id`
            identity stable across re-uploads regardless of the temp
            filename any single request happened to stream through.

        Returns
        -------
        dict[str, Any]
            ``{"document_id", "chunks_written", "changed"}``.

        Raises
        ------
        ValueError
            If `path`'s extension isn't in `config.ingestion.supported_extensions`.
        IngestGovernanceError
            If `caller` is set and the loaded document's `tenant_id` names a
            different tenant than `caller.tenant_id` without `caller`
            holding a cross-tenant support role. See
            `rag.ingestion.governance.resolve_ingest_tenant_id`.
        """
        path = Path(path)
        if path.suffix.lower() not in self._config.ingestion.supported_extensions:
            raise ValueError(f"Unsupported extension '{path.suffix}' for {path}")

        raw_document = get_loader(path).load(path)
        updates: dict[str, Any] = {}
        if source_override is not None:
            updates["source"] = source_override
        if caller is not None:
            updates["tenant_id"] = resolve_ingest_tenant_id(raw_document.tenant_id, caller)
        if updates:
            raw_document = raw_document.model_copy(update=updates)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        document_id, changed = self._vectorstore.get_or_create_document_id(
            source=raw_document.source, checksum=checksum, dataset_id=dataset_id
        )

        if not changed:
            logger.info(
                "skip_unchanged_document",
                extra={
                    "source": raw_document.source,
                    "document_id": document_id,
                    "dataset_id": dataset_id,
                },
            )
            return {"document_id": document_id, "chunks_written": 0, "changed": False}

        self._vectorstore.delete_chunks_by_document_id(document_id)

        cleaned = self._cleaner.clean(raw_document.content)
        chunk_spans = self._chunker.split(cleaned, source_type=raw_document.source_type)
        chunks = self._writer.write(
            raw_document, document_id, chunk_spans, dataset_id=dataset_id, category=category
        )

        logger.info(
            "ingested_document",
            extra={
                "source": raw_document.source,
                "document_id": document_id,
                "chunk_count": len(chunks),
                "category": category,
                "dataset_id": dataset_id,
            },
        )
        return {"document_id": document_id, "chunks_written": len(chunks), "changed": True}

    def ingest_path(self, path: Path, dataset_id: str) -> IngestionStats:
        """Ingest a single file, or recursively ingest a directory tree.

        Every chunk written is tagged with `dataset_id`, isolating it from
        every other dataset at retrieval time. When walking a directory,
        each file's path relative to `path` is additionally recorded as
        the chunk metadata's `category`.

        For a directory target, also detects documents deleted since the
        last ingestion (diffing the pre-run set of known sources against
        what's discovered this run) and deletes them via
        `delete_documents_by_source`. Skipped for a single-file target.

        Parameters
        ----------
        path : Path
            A single file, or a directory to walk recursively.
        dataset_id : str
            Namespace tag stored on every chunk.

        Returns
        -------
        IngestionStats
            Aggregate counts plus one `ingest_file`-shaped entry (with an
            added `"status"` key: `"new"`/`"changed"`/`"unchanged"`) per
            file processed, in `.results`.
        """
        path = Path(path)
        existing_sources = set(self._vectorstore.list_document_sources(dataset_id))

        if not path.is_dir():
            result = self.ingest_file(path, dataset_id)
            discovered = [str(path)]
            return self._aggregate_ingestion_stats(
                [result], discovered, dataset_id, existing_sources, detect_deletions=False
            )

        results = []
        discovered = []
        for ext in self._config.ingestion.supported_extensions:
            for file_path in sorted(path.rglob(f"*{ext}")):
                relative_dir = file_path.relative_to(path).parent
                category = None if relative_dir == Path(".") else relative_dir.as_posix()
                results.append(self.ingest_file(file_path, dataset_id, category=category))
                discovered.append(str(file_path))
        return self._aggregate_ingestion_stats(
            results, discovered, dataset_id, existing_sources, detect_deletions=True
        )

    def _aggregate_ingestion_stats(
        self,
        results: list[dict[str, Any]],
        discovered_sources: list[str],
        dataset_id: str,
        existing_sources_before: set[str],
        detect_deletions: bool,
    ) -> IngestionStats:
        """Classify each file's status, delete removed documents, and total chunk counts."""
        deleted_count = 0
        if detect_deletions:
            deleted_sources = existing_sources_before - set(discovered_sources)
            if deleted_sources:
                deleted_count = self._vectorstore.delete_documents_by_source(
                    dataset_id, sorted(deleted_sources)
                )

        new_count = changed_count = unchanged_count = 0
        chunks_embedded = 0
        reused_document_ids: list[str] = []
        for source, result in zip(discovered_sources, results, strict=True):
            if not result["changed"]:
                status = "unchanged"
                unchanged_count += 1
                reused_document_ids.append(result["document_id"])
            elif source in existing_sources_before:
                status = "changed"
                changed_count += 1
                chunks_embedded += result["chunks_written"]
            else:
                status = "new"
                new_count += 1
                chunks_embedded += result["chunks_written"]
            result["status"] = status

        chunks_reused = 0
        if reused_document_ids:
            counts = self._vectorstore.count_chunks_by_document(dataset_id)
            chunks_reused = sum(counts.get(doc_id, 0) for doc_id in reused_document_ids)

        return IngestionStats(
            discovered=len(discovered_sources),
            new=new_count,
            changed=changed_count,
            unchanged=unchanged_count,
            deleted=deleted_count,
            chunks_embedded=chunks_embedded,
            chunks_reused=chunks_reused,
            results=results,
            vision_stats=self._writer.vision_stats if self._writer.vision_provider else None,
        )


def main() -> None:
    """CLI entrypoint: parse args and run `IngestionPipeline.ingest_path`."""
    from rag.logging_config import configure_logging

    parser = argparse.ArgumentParser(description="Ingest a file or directory into the vector store")
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Namespace tag stored on every chunk (e.g. 'techfusion'). Required; "
        "isolates this ingestion from every other dataset at retrieval/eval time.",
    )
    parser.add_argument("--config", default=None, help="Override config/default.yaml")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete every existing document/chunk tagged with --dataset-id before ingesting.",
    )
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    configure_logging(config.app.log_level)

    pipeline = IngestionPipeline(config)
    if args.clear:
        pipeline.clear_dataset(args.dataset_id)
        print(f"Cleared dataset '{args.dataset_id}'")
    stats = pipeline.ingest_path(Path(args.path), args.dataset_id)
    for result in stats.results:
        print(result)
    print(
        f"discovered={stats.discovered} new={stats.new} changed={stats.changed} "
        f"unchanged={stats.unchanged} deleted={stats.deleted} "
        f"chunks_embedded={stats.chunks_embedded} chunks_reused={stats.chunks_reused}"
    )
    if stats.vision_stats is not None:
        v = stats.vision_stats
        avg_ms = v.total_latency_ms / v.cache_misses if v.cache_misses else 0.0
        print(
            f"vision: images_processed={v.images_processed} cache_hits={v.cache_hits} "
            f"cache_misses={v.cache_misses} avg_call_latency_ms={avg_ms:.1f}"
        )


if __name__ == "__main__":
    main()
