# Multimodal & Layout-Aware Ingestion

PDF and DOCX documents keep their real structure (headings, tables,
code/config blocks, images, page numbers) instead of being flattened to
plain text: both loaders serialize what they extract into the same
Markdown-equivalent syntax the `structured_markdown` chunker already
parses, rather than introducing a second, parallel element model.

An optional local vision provider (`config.vision.provider: "ollama"`,
default `"none"`) can describe an image with a small offline model
(`moondream` by default). A vision description is always a second,
sibling chunk alongside the caption/alt-text chunk, never overwriting it.

Full design writeups:

- [Multimodal + Relationship-Aware Ingestion](../architecture.md#multimodal-relationship-aware-ingestion)
- [Layout-Aware Document Ingestion and Vision](../architecture.md#layout-aware-document-ingestion-and-vision)
- API reference: [Vision](../reference/vision.md), [Loaders](../reference/loaders.md), [Chunkers](../reference/chunkers.md)
