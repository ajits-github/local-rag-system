"""MCP-facing serialization of retrieval results.

`McpChunkResult` mirrors `SourceItem`'s public citation shape
(`rag.api.routers.query`), plus the chunk's own already-authorized,
already-sanitized `content` (an MCP caller has no separate answer-
synthesis step and must read the retrieved text directly). It never
exposes `tenant_id`/`trust_level`/`classification`/`sensitive_field_ids`
or the raw embedding vector, matching what `SourceItem` already treats
as safe for an external caller.
"""

from __future__ import annotations

from pydantic import BaseModel

from rag.schemas import SearchResult


class McpChunkResult(BaseModel):
    """One retrieved chunk, already authorized, freshness-filtered, and sanitized.

    Attributes
    ----------
    chunk_id, document_id, source : str
        Identifiers and the dataset-root-relative source path.
    content : str
        The chunk's text, after field redaction. Never the raw,
        unredacted value.
    category, content_type, section_path, page, attachment_name,
    source_anchor : optional
        Structural metadata, same fields `SourceItem` exposes over
        `POST /query`/`POST /agent/query`.
    score : float
        Similarity/ranking score; meaningless for a `tool_fetched`/
        `expanded` origin, which inherit their originating result's
        score.
    origin : str
        `"retrieved"`, `"expanded"`, or `"tool_fetched"`.
    vision_generated : bool
        Whether this chunk's content is a vision-model description.
    injection_suspected : bool
        Whether `detect_injection` flagged this chunk's content.
        Observability only; the calling model should weigh it, not treat
        it as a directive.
    """

    chunk_id: str
    document_id: str
    source: str
    content: str
    category: str | None = None
    score: float
    origin: str
    content_type: str | None = None
    section_path: str | None = None
    page: int | None = None
    attachment_name: str | None = None
    source_anchor: str | None = None
    vision_generated: bool = False
    injection_suspected: bool = False


def to_mcp_result(result: SearchResult) -> McpChunkResult:
    """Convert one already-sanitized `SearchResult` into its MCP-facing shape."""
    meta = result.chunk.metadata
    return McpChunkResult(
        chunk_id=meta.chunk_id,
        document_id=meta.document_id,
        source=meta.source,
        content=result.chunk.content,
        category=meta.category,
        score=result.score,
        origin=result.origin,
        content_type=meta.content_type,
        section_path=meta.section_path,
        page=meta.page,
        attachment_name=meta.attachment_name,
        source_anchor=meta.source_anchor,
        vision_generated=meta.vision_generated,
        injection_suspected=result.injection_suspected,
    )
