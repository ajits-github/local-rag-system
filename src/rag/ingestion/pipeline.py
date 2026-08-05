"""Explicit ingestion pipeline: loader -> cleaner -> chunker -> embedder -> writer.

Also runnable as a CLI: `python -m rag.ingestion.pipeline <path>` (used by
`make ingest FILE=...`).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

from rag.chunkers.registry import get_chunker
from rag.cleaners.default_cleaner import DefaultCleaner
from rag.config import AppConfig, load_config
from rag.embedders.base import Embedder
from rag.factory import build_embedder, build_vectorstore
from rag.ingestion.writer import Writer
from rag.loaders.registry import get_loader
from rag.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        config: AppConfig,
        vectorstore: VectorStore | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._config = config
        self._vectorstore = vectorstore or build_vectorstore(config)
        self._embedder = embedder or build_embedder(config)
        self._cleaner = DefaultCleaner()
        self._chunker = get_chunker(config.chunking)
        self._writer = Writer(self._embedder, self._vectorstore)

    def ingest_file(self, path: Path, category: str | None = None) -> dict:
        path = Path(path)
        if path.suffix.lower() not in self._config.ingestion.supported_extensions:
            raise ValueError(f"Unsupported extension '{path.suffix}' for {path}")

        raw_document = get_loader(path).load(path)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        document_id, changed = self._vectorstore.get_or_create_document_id(
            source=raw_document.source, checksum=checksum
        )

        if not changed:
            logger.info(
                "skip_unchanged_document",
                extra={"source": raw_document.source, "document_id": document_id},
            )
            return {"document_id": document_id, "chunks_written": 0, "changed": False}

        self._vectorstore.delete_chunks_by_document_id(document_id)

        cleaned = self._cleaner.clean(raw_document.content)
        chunk_texts = self._chunker.split(cleaned)
        chunks = self._writer.write(raw_document, document_id, chunk_texts, category=category)

        logger.info(
            "ingested_document",
            extra={
                "source": raw_document.source,
                "document_id": document_id,
                "chunk_count": len(chunks),
                "category": category,
            },
        )
        return {"document_id": document_id, "chunks_written": len(chunks), "changed": True}

    def ingest_path(self, path: Path) -> list[dict]:
        """Ingest a single file, or recursively ingest a directory tree.

        When walking a directory, each file's path relative to `path` (minus
        its own filename) is recorded as the chunk metadata's `category` —
        e.g. ingesting `data/knowledge_base` preserves `security`,
        `runbooks/postgres`, etc. as filterable metadata.
        """
        path = Path(path)
        if not path.is_dir():
            return [self.ingest_file(path)]

        results = []
        for ext in self._config.ingestion.supported_extensions:
            for file_path in sorted(path.rglob(f"*{ext}")):
                relative_dir = file_path.relative_to(path).parent
                category = None if relative_dir == Path(".") else relative_dir.as_posix()
                results.append(self.ingest_file(file_path, category=category))
        return results


def main() -> None:
    from rag.logging_config import configure_logging

    parser = argparse.ArgumentParser(description="Ingest a file or directory into the vector store")
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument("--config", default=None, help="Override config/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    configure_logging(config.app.log_level)

    pipeline = IngestionPipeline(config)
    for result in pipeline.ingest_path(Path(args.path)):
        print(result)


if __name__ == "__main__":
    main()
