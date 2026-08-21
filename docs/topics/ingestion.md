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
