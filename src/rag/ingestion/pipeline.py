"""Explicit ingestion pipeline: loader -> cleaner -> chunker -> embedder -> writer.

Also runnable as a CLI: `python -m rag.ingestion.pipeline <path> --dataset-id <id>`
(used by `make ingest FILE=... DATASET_ID=...`).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from rag.chunkers.registry import get_chunker
from rag.cleaners.default_cleaner import DefaultCleaner
from rag.config import AppConfig, load_config
from rag.embedders.base import Embedder
from rag.factory import (
    build_embedder,
    build_graph_client,
    build_llm,
    build_vectorstore,
    build_vision_provider,
)
from rag.graph.client import Neo4jClient
from rag.graph.extraction import GraphExtractionStats, extract_and_write_for_document
from rag.graph.upsert import GraphWriter
from rag.ingestion.governance import (
    IngestCallerContext,
    IngestGovernanceError,
    resolve_ingest_tenant_id,
)
from rag.ingestion.writer import Writer
from rag.loaders.registry import get_loader
from rag.path_matching import source_matches_relevant
from rag.prompts.loader import load_prompt_template
from rag.schemas import Chunk, IngestionStats, RawDocument
from rag.vectorstore.base import VectorStore
from rag.vision.base import VisionProvider

logger = logging.getLogger(__name__)

# A Markdown image reference's target, as the structured chunker reads it.
_IMAGE_REFERENCE_RE = re.compile(r"!\[[^\]\[]*\]\(([^()\s]+)\)")


def _source_is_under_root(source: str, root: Path) -> bool:
    """Return whether a persisted `source` path resolves to `root` or one of its descendants.

    Limits deletion detection to documents that could have come from this
    root. A document ingested from another root is never treated as deleted.
    Matching compares resolved paths, not string prefixes, so `knowledge_base2`
    is not a child of `knowledge_base`.

    Parameters
    ----------
    source : str
        A persisted document's `source` value.
    root : Path
        The directory currently being walked by `ingest_path`.

    Returns
    -------
    bool
        True if `source` lies at or under `root`.
    """
    try:
        resolved_source = Path(source.replace("\\", "/")).resolve()
        resolved_root = root.resolve()
    except OSError:
        return False
    return resolved_source.is_relative_to(resolved_root)


class IngestionPipeline:
    """Runs a document through loader -> cleaner -> chunker -> embedder -> writer."""

    def __init__(
        self,
        config: AppConfig,
        vectorstore: VectorStore | None = None,
        embedder: Embedder | None = None,
        vision_provider: VisionProvider | None = None,
        graph_client: Neo4jClient | None = None,
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
            `config.vision.provider` if omitted (`None` when the
            configured provider is `"none"`). Inject a test mock here to
            exercise vision-mode ingestion without config changes.
        graph_client : Neo4jClient | None, optional
            Graph client for the optional graph-ingestion stage (see
            `_run_graph_extraction`); built from `config.graph.enabled`
            if omitted (`None` when disabled). Inject a test double or
            `None` to exercise ingestion without a real Neo4j instance.
        """
        self._config = config
        self._vectorstore = vectorstore or build_vectorstore(config)
        self._embedder = embedder or build_embedder(config)
        self._cleaner = DefaultCleaner()
        self._chunker = get_chunker(config.chunking)
        self._writer = Writer(
            self._embedder, self._vectorstore, vision_provider or build_vision_provider(config)
        )
        self._graph_client = (
            graph_client if graph_client is not None else build_graph_client(config)
        )
        self._graph_writer = (
            GraphWriter(self._graph_client) if self._graph_client is not None else None
        )
        self._graph_llm = None
        self._graph_extraction_template = None
        if (
            self._graph_client is not None
            and config.graph.extraction.enabled
            and config.graph.extraction.llm_extraction_enabled
        ):
            self._graph_llm = build_llm(config)
            self._graph_extraction_template = load_prompt_template(
                str(
                    config.agent_prompt_template_path(
                        config.graph.extraction.extraction_prompt_path
                    )
                )
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
        if self._graph_writer is not None:
            try:
                self._graph_writer.delete_facts_by_dataset(dataset_id)
            except Exception:
                logger.warning(
                    "Failed to clear graph facts for dataset %s", dataset_id, exc_info=True
                )

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
            (the default) for the CLI/`ingest_path`, which never supply
            one, and for `POST /ingest` when JWT auth is disabled or an
            unauthenticated request was allowed through
            (`insecure_dev_mode`); in that case the loaded document's own
            `tenant_id` (front matter, or `None`) is persisted as parsed.
            When set, `resolve_ingest_tenant_id` resolves the effective
            `tenant_id` before any database write happens, so a rejected
            upload leaves no trace.
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
            If `caller` is set and either: the loaded document's
            `tenant_id` names a different tenant than `caller.tenant_id`
            without `caller` holding a cross-tenant support role (see
            `rag.ingestion.governance.resolve_ingest_tenant_id`); or this
            `(source, dataset_id)` already resolves to an existing
            document owned by a different tenant's chunks, which would
            otherwise let a caller's own, legitimate ingest permission
            silently replace another tenant's content just because a
            same-named upload happened to resolve to the same
            `document_id`.
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
        checksum = self._document_checksum(path, raw_document)
        document_id, changed = self._vectorstore.get_or_create_document_id(
            source=raw_document.source, checksum=checksum, dataset_id=dataset_id
        )

        if changed and caller is not None:
            # A legitimate ingest permission under one's OWN tenant must
            # never become a cross-tenant destructive write just because
            # another tenant's upload happened to resolve to the same
            # (source, dataset_id) identity (e.g. the same filename shared
            # across tenants in one dataset). Checked against the existing
            # chunks' own tenant_id, not just the new content's resolved
            # tenant_id, since governance.resolve_ingest_tenant_id only
            # ever validated the latter.
            existing_tenant_id = self._vectorstore.get_document_owner_tenant_id(document_id)
            if (
                existing_tenant_id is not None
                and existing_tenant_id != raw_document.tenant_id
                and not caller.is_privileged
            ):
                raise IngestGovernanceError(
                    f"document_id={document_id!r} (source={raw_document.source!r}) already "
                    f"belongs to tenant_id={existing_tenant_id!r}, which does not match the "
                    "authenticated caller's own tenant, and the caller does not hold a "
                    "cross-tenant support role."
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

        cleaned = self._cleaner.clean(raw_document.content)
        chunk_spans = self._chunker.split(cleaned, source_type=raw_document.source_type)
        chunks = self._writer.write(
            raw_document,
            document_id,
            chunk_spans,
            dataset_id=dataset_id,
            category=category,
            asset_base_dir=path.parent,
        )
        # One transaction: the checksum and the chunk replacement commit together,
        # so a failure cannot leave the checksum ahead of the stored chunks.
        self._vectorstore.replace_document_chunks(document_id, checksum, chunks)

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
        # Runs after the Postgres commit. Postgres is the source of truth, and a
        # graph failure never fails this ingestion.
        self._run_graph_extraction(
            raw_document, document_id, dataset_id, category, checksum, chunks
        )
        return {"document_id": document_id, "chunks_written": len(chunks), "changed": True}

    def _document_checksum(self, path: Path, raw_document: RawDocument) -> str:
        """Return the change-detection checksum for `path`.

        The sha256 of the file's bytes. When a vision provider is configured,
        each referenced local image's digest (or `missing`) is folded in too,
        so replacing or removing an image re-ingests the document and drops
        its stale vision chunk; a document with no image references keeps
        its plain-bytes checksum.
        """
        digest = hashlib.sha256(path.read_bytes())
        if self._writer.vision_provider is None:
            return digest.hexdigest()
        for anchor in sorted(set(_IMAGE_REFERENCE_RE.findall(raw_document.content))):
            if "://" in anchor:
                continue
            asset = (path.parent / anchor).resolve()
            asset_digest = (
                hashlib.sha256(asset.read_bytes()).hexdigest() if asset.is_file() else "missing"
            )
            digest.update(f"|{anchor}|{asset_digest}".encode())
        return digest.hexdigest()

    def _run_graph_extraction(
        self,
        raw_document: RawDocument,
        document_id: str,
        dataset_id: str,
        category: str | None,
        checksum: str,
        chunks: list[Chunk],
    ) -> GraphExtractionStats | None:
        """Run graph extraction for one document. Returns `None` when graph extraction is disabled.

        Resolves `supersedes_document_id` with the path-suffix rule that
        `retrieval/freshness.py` uses, then delegates to
        `extract_and_write_for_document`. Any exception is logged and
        swallowed, so a graph failure never fails the ingestion.

        Returns
        -------
        GraphExtractionStats | None
            `None` when graph extraction is disabled, otherwise the stats
            from `extract_and_write_for_document`.
        """
        if not (self._config.graph.enabled and self._config.graph.extraction.enabled):
            return None
        if self._graph_client is None or self._graph_writer is None:
            return None
        try:
            supersedes_document_id = None
            if raw_document.supersedes_source:
                versions = self._vectorstore.list_document_versions(dataset_id)
                supersedes_document_id = next(
                    (
                        v.document_id
                        for v in versions
                        if source_matches_relevant(v.source, raw_document.supersedes_source)
                    ),
                    None,
                )
            return extract_and_write_for_document(
                self._graph_writer,
                self._graph_client,
                self._config,
                self._graph_llm,
                self._graph_extraction_template,
                document_id=document_id,
                dataset_id=dataset_id,
                tenant_id=raw_document.tenant_id,
                source=raw_document.source,
                title=raw_document.title,
                category=category,
                classification=raw_document.classification,
                status=raw_document.status,
                document_version=raw_document.document_version,
                trust_level=raw_document.trust_level,
                allowed_roles=raw_document.allowed_roles,
                checksum=checksum,
                supersedes_document_id=supersedes_document_id,
                chunks=chunks,
            )
        except Exception:
            logger.warning("Graph extraction failed for %s", document_id, exc_info=True)
            return None

    def ingest_path(self, path: Path, dataset_id: str) -> IngestionStats:
        """Ingest a single file, or recursively ingest a directory tree.

        Every chunk is tagged with `dataset_id`; walking a directory also
        tags each chunk's `category` from the file's path relative to
        `path`.

        For a directory target only, also detects documents removed since
        the last ingestion of this root (see `_source_is_under_root`) and
        deletes them via `delete_documents_by_source`.

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
        scoped_existing_sources = {
            source for source in existing_sources if _source_is_under_root(source, path)
        }
        return self._aggregate_ingestion_stats(
            results, discovered, dataset_id, scoped_existing_sources, detect_deletions=True
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
                deleted_document_ids: list[str] = []
                if self._graph_writer is not None:
                    # Resolved before the Postgres delete, which returns only a count.
                    versions = self._vectorstore.list_document_versions(dataset_id)
                    deleted_document_ids = [
                        v.document_id for v in versions if v.source in deleted_sources
                    ]
                deleted_count = self._vectorstore.delete_documents_by_source(
                    dataset_id, sorted(deleted_sources)
                )
                if self._graph_writer is not None:
                    for document_id in deleted_document_ids:
                        try:
                            self._graph_writer.delete_facts_by_document(document_id)
                        except Exception:
                            logger.warning(
                                "Failed to delete graph facts for removed document %s",
                                document_id,
                                exc_info=True,
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
