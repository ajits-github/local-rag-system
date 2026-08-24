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

function badgeClass(contentType: string | null | undefined): string {
  const key = contentType ?? "prose";
  return `source-badge source-badge--${key}`;
}

function SourceRow({ source }: { source: SourceItem }) {
  const label = CONTENT_TYPE_LABEL[source.content_type ?? "prose"] ?? source.content_type ?? "Text";
  return (
    <li className="source-row">
      <div className="source-row__header">
        <span className={badgeClass(source.content_type)}>{label}</span>
        <span className="source-row__name" title={source.source}>
          {source.source}
        </span>
        <span className="source-row__score">{source.score.toFixed(3)}</span>
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

export function SourcesPanel({ sources }: { sources: SourceItem[] }) {
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
            <SourceRow key={source.chunk_id} source={source} />
          ))}
        </ul>
      )}
    </div>
  );
}
