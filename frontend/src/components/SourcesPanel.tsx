import { useState } from "react";
import type { SourceItem } from "../api/types";

const CONTENT_TYPE_LABEL: Record<string, string> = {
  prose: "Text",
  table: "Table",
  code: "Code",
  configuration: "Config",
  image: "Image",
  chart: "Chart",
};

const ORIGIN_LABEL: Record<SourceItem["origin"], string> = {
  retrieved: "RAG",
  expanded: "RAG (related)",
  tool_fetched: "RAG (tool)",
  mcp_remote: "MCP remote",
};

function badgeClass(contentType: string | null | undefined): string {
  const key = contentType ?? "prose";
  return `source-badge source-badge--${key}`;
}

function SourceRow({ source, showScore }: { source: SourceItem; showScore: boolean }) {
  const label = CONTENT_TYPE_LABEL[source.content_type ?? "prose"] ?? source.content_type ?? "Text";
  const isMcp = source.origin === "mcp_remote";
  return (
    <li className="source-row">
      <div className="source-row__header">
        <span className={badgeClass(source.content_type)}>{label}</span>
        <span
          className={`origin-badge ${isMcp ? "origin-badge--mcp" : "origin-badge--rag"}`}
          title={isMcp ? "Sourced from an MCP remote/business tool, not the local knowledge base" : "Sourced from the local RAG knowledge base"}
        >
          {ORIGIN_LABEL[source.origin]}
        </span>
        <span className="source-row__name" title={source.source}>
          {source.source}
        </span>
        {showScore && <span className="source-row__score">{source.score.toFixed(3)}</span>}
      </div>
      <div className="source-row__meta">
        {source.section_path && <span>{source.section_path}</span>}
        {source.page != null && <span>page {source.page}</span>}
        {source.category && <span>{source.category}</span>}
        {source.attachment_name && <span>attachment: {source.attachment_name}</span>}
        {source.vision_generated && <span className="source-row__vision">vision-described</span>}
      </div>
    </li>
  );
}

/**
 * `showScore` defaults to true: the raw relevance score is not a secret,
 * just internal-debug-flavored detail (see CLAUDE.md's "production vs.
 * developer UI mode" note) -- callers that want it hidden in production
 * mode pass `showScore={false}` explicitly (see MessageBubble).
 */
export function SourcesPanel({ sources, showScore = true }: { sources: SourceItem[]; showScore?: boolean }) {
  const [expanded, setExpanded] = useState(false);
  if (sources.length === 0) return null;

  return (
    <div className="collapsible-panel">
      <button
        type="button"
        className="collapsible-panel__toggle"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
      >
        {expanded ? "▾" : "▸"} Sources ({sources.length})
      </button>
      {expanded && (
        <ul className="source-list">
          {sources.map((source) => (
            <SourceRow key={source.chunk_id} source={source} showScore={showScore} />
          ))}
        </ul>
      )}
    </div>
  );
}
