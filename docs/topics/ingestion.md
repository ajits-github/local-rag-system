# Ingestion & Chunking

Documents are loaded per source type (Markdown, plain text, HTML, PDF,
DOCX), normalized by a cleaner, then split into chunks by a chunker. The
default chunker, `structured_markdown`, keeps tables, fenced code/config
blocks, and standalone images atomic instead of letting a naive
recursive-character splitter cut through them, and tags each chunk with
structural metadata (`content_type`, `section_path`, `page`,
`parent_chunk_id`) rather than treating a document as one flat string.

Ingestion is incremental and idempotent: re-ingesting an unchanged file is
a no-op (checksum-based), an edited file is re-chunked under its existing
`document_id`, and documents removed from a directory target are detected
and deleted from the vector store on the next run.

For the full design (element model, relationship linkage between a table/
image and its nearest preceding prose, layout-aware PDF/DOCX extraction),
see:

- [Multimodal + Relationship-Aware Ingestion](../architecture.md#multimodal-relationship-aware-ingestion)
- [Layout-Aware Document Ingestion and Vision](../architecture.md#layout-aware-document-ingestion-and-vision)
- API reference: [Loaders](../reference/loaders.md), [Chunkers](../reference/chunkers.md), [Ingestion](../reference/ingestion.md)

## Metadata: category, dataset_id, and structured-content tags

`ChunkMetadata.category` is the ingested file's folder path relative to
whatever root `ingest_path()` walked (e.g. `security`, or `security/sub`
for nested folders); `None` for a single-file ingest or an API upload,
which have no folder context. Derived in `ingestion/pipeline.py`'s
`ingest_path()`, not by any loader.

Beyond `category`/`dataset_id`, `ChunkMetadata` also carries structural
hints originating from `ChunkSpan` (`schemas.py`): `content_type`
(`"prose"` default, or `"table"`/`"code"`/`"configuration"`/`"chart"`),
`section_path` (the Markdown heading breadcrumb, e.g. `"Setup > Docker"`),
`code_language`, `table_headers`, and `attachment_name`/`source_anchor`
(local asset links, e.g. an embedded `.svg`/`.png`/`.pdf`). Only the
`structured_markdown` chunker (`chunkers/structured_markdown.py`, the
default `chunking.provider`) populates these beyond the prose default,
it walks Markdown line-by-line keeping fenced code/config blocks and
tables atomic instead of letting `RecursiveCharacterChunker` slice through
them, and tags chart-like fenced blocks (a ` ```text ` fence immediately
followed by an emphasis-wrapped caption paragraph) as `content_type:
chart`. `Writer.write` backfills `content_type="prose"` for any span that
leaves it unset, so a persisted chunk's `content_type` is never `None`.
`eval/content_type.py` re-runs this same chunker over gold-referenced
documents to bucket eval questions by structured-content type (table /
code_configuration / chart / prose / multi_hop / unanswerable) for
`eval/report_content_type.py`'s breakdown report, it reuses the real
loader→cleaner→chunker chain rather than a separate classifier, so it
stays correct as chunking logic evolves.

`category`, `dataset_id`, and `content_type` are all in
`VectorStore.base`'s `ALLOWED_FILTER_FIELDS` (an explicit whitelist, so a
`filters` dict can never inject an arbitrary SQL column), so `POST
/query`'s `filters` can restrict retrieval to any of them directly.

For Markdown with a YAML front-matter block (`title`/`owner`/
`last_reviewed`), `loaders/text_loader.py` parses it for
`title`/`author`/`last_modified` and strips the block before chunking;
plain `.txt` or front-matter-less `.md` fall back to filename/filesystem
timestamps as before.
