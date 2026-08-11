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
from rag.ingestion.writer import Writer
from rag.loaders.registry import get_loader
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
            only provider implemented so far -- so this is `None` in
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
        self, path: Path, dataset_id: str, category: str | None = None
    ) -> dict[str, Any]:
        """Load, clean, chunk, embed, and persist a single file.

        Skips re-writing chunks if the file's checksum is unchanged since
        the last ingestion of the same (source, dataset_id).

        Parameters
        ----------
        path : Path
            File to ingest.
        dataset_id : str
            Namespace tag stored on every chunk.
        category : str | None, optional
            Folder-derived category tag, by default None.

        Returns
        -------
        dict[str, Any]
            ``{"document_id", "chunks_written", "changed"}``.

        Raises
        ------
        ValueError
            If `path`'s extension isn't in `config.ingestion.supported_extensions`.
        """
        path = Path(path)
        if path.suffix.lower() not in self._config.ingestion.supported_extensions:
            raise ValueError(f"Unsupported extension '{path.suffix}' for {path}")

        raw_document = get_loader(path).load(path)
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

    def ingest_path(self, path: Path, dataset_id: str) -> list[dict[str, Any]]:
        """Ingest a single file, or recursively ingest a directory tree.

        Every chunk written is tagged with `dataset_id`, which isolates it
        from every other dataset at retrieval time (see
        `vectorstore.base.ALLOWED_FILTER_FIELDS` and `eval/run_eval.py`,
        which filters on it unconditionally). When walking a directory,
        each file's path relative to `path` (minus its own filename) is
        additionally recorded as the chunk metadata's `category` — e.g.
        ingesting `data/knowledge_base` preserves `security`,
        `runbooks/postgres`, etc. as filterable metadata within that
        dataset.

        Parameters
        ----------
        path : Path
            A single file, or a directory to walk recursively.
        dataset_id : str
            Namespace tag stored on every chunk.

        Returns
        -------
        list[dict[str, Any]]
            One `ingest_file` result dict per file ingested.
        """
        path = Path(path)
        if not path.is_dir():
            return [self.ingest_file(path, dataset_id)]

        results = []
        for ext in self._config.ingestion.supported_extensions:
            for file_path in sorted(path.rglob(f"*{ext}")):
                relative_dir = file_path.relative_to(path).parent
                category = None if relative_dir == Path(".") else relative_dir.as_posix()
                results.append(self.ingest_file(file_path, dataset_id, category=category))
        return results


def main() -> None:
    """CLI entrypoint: parse args and run `IngestionPipeline.ingest_path`."""
    from rag.logging_config import configure_logging

    parser = argparse.ArgumentParser(description="Ingest a file or directory into the vector store")
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Namespace tag stored on every chunk (e.g. 'techfusion'). Required -- "
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
    for result in pipeline.ingest_path(Path(args.path), args.dataset_id):
        print(result)


if __name__ == "__main__":
    main()
