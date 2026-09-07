import { useState } from "react";
import { useRuntimeInfo } from "../hooks/useRuntimeInfo";
import type { RuntimeInfo } from "../api/types";

function boolLabel(value: boolean): string {
  return value ? "enabled" : "disabled";
}

/** Short "model · retrieval mode · fusion · MCP" summary for the collapsed toggle. */
function compactBadge(info: RuntimeInfo): string {
  const parts = [info.generation_model, info.retrieval_mode === "hybrid" ? "Hybrid" : "Dense"];
  if (info.fusion_method) parts.push(info.fusion_method.toUpperCase());
  if (info.mcp_client_enabled || info.mcp_enabled) parts.push("MCP");
  return parts.join(" · ");
}

/**
 * Collapsible "Runtime configuration" panel: system-wide pipeline info
 * (models, retrieval mode, fusion, reranker, MCP/agent/vision/security
 * toggles) fetched from GET /info. Deliberately distinct from DebugPanel,
 * which shows *per-answer* data (route, latency, tool calls) for one
 * message -- this panel shows the same values for every message, since
 * they describe the backend process, not a single request.
 *
 * The collapsed toggle itself doubles as a compact badge (e.g.
 * "qwen2.5:3b · Hybrid · RRF · MCP") once loaded, so the essentials are
 * visible without expanding; the full field-by-field breakdown stays
 * behind the toggle.
 */
export function RuntimeInfoPanel() {
  const [expanded, setExpanded] = useState(false);
  const { info, loading, error, load } = useRuntimeInfo();

  return (
    <div className="collapsible-panel collapsible-panel--runtime">
      <button
        type="button"
        className="collapsible-panel__toggle"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
      >
        {expanded ? "▾" : "▸"} {info ? `Runtime: ${compactBadge(info)}` : "Runtime configuration"}
        {error && !loading && (
          <span className="runtime-info__badge-error" role="status">
            {" "}
            (unavailable)
          </span>
        )}
      </button>
      {expanded && (
        <div className="runtime-info">
          {loading && <span className="runtime-info__status">Loading runtime configuration…</span>}
          {error && !loading && (
            <span className="runtime-info__status runtime-info__status--error" role="status">
              Runtime configuration unavailable
            </span>
          )}
          {info && !loading && !error && (
            <dl className="debug-grid">
              <dt>Generation</dt>
              <dd>
                {info.generation_provider} / {info.generation_model}
              </dd>
              <dt>Embedding model</dt>
              <dd>{info.embedding_model}</dd>
              <dt>Vector store</dt>
              <dd>{info.vectorstore_provider}</dd>
              <dt>Retrieval mode</dt>
              <dd>{info.retrieval_mode}</dd>
              <dt>Sparse retrieval (BM25)</dt>
              <dd>{boolLabel(info.sparse_retrieval_enabled)}</dd>
              <dt>Fusion method</dt>
              <dd>{info.fusion_method ?? "none"}</dd>
              <dt>Reranker</dt>
              <dd>
                {info.reranker_provider} ({boolLabel(info.reranker_enabled)})
              </dd>
              <dt>Agent</dt>
              <dd>{boolLabel(info.agent_enabled)}</dd>
              <dt>MCP server</dt>
              <dd>{boolLabel(info.mcp_enabled)}</dd>
              <dt>MCP client</dt>
              <dd>{boolLabel(info.mcp_client_enabled)}</dd>
              <dt>Vision provider</dt>
              <dd>{info.vision_provider}</dd>
              <dt>Tracing</dt>
              <dd>{boolLabel(info.tracing_enabled)}</dd>
              <dt>Rate limiting</dt>
              <dd>{boolLabel(info.rate_limit_enabled)}</dd>
              <dt>Field redaction</dt>
              <dd>{boolLabel(info.field_redaction_enabled)}</dd>
              <dt>Authorization</dt>
              <dd>{boolLabel(info.authorization_enabled)}</dd>
              <dt>JWT auth</dt>
              <dd>{boolLabel(info.auth_enabled)}</dd>
            </dl>
          )}
          <button type="button" className="runtime-info__refresh" onClick={load} disabled={loading}>
            Refresh
          </button>
        </div>
      )}
    </div>
  );
}
