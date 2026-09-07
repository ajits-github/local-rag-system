import { useState } from "react";
import type { ToolCallDetail } from "../api/types";
import type { DebugInfo } from "../state/types";

const EXECUTION_LABEL: Record<ToolCallDetail["execution"], string> = {
  local: "Local",
  mcp_remote: "MCP remote",
};

function ToolCallCard({ detail }: { detail: ToolCallDetail }) {
  return (
    <li className={`tool-call-card ${detail.success ? "tool-call-card--success" : "tool-call-card--failed"}`}>
      <span className="tool-call-card__name">{detail.tool_name}</span>
      <dl className="tool-call-card__meta">
        <dt>Execution</dt>
        <dd>{EXECUTION_LABEL[detail.execution]}</dd>
        <dt>Status</dt>
        <dd>{detail.success ? "Success" : "Failed"}</dd>
        <dt>Duration</dt>
        <dd>{Math.round(detail.latency_ms)} ms</dd>
      </dl>
    </li>
  );
}

export function DebugPanel({ debug }: { debug: DebugInfo | undefined }) {
  const [expanded, setExpanded] = useState(false);
  if (!debug) return null;

  const hasToolDetails = Boolean(debug.toolCallDetails && debug.toolCallDetails.length > 0);
  const hasToolNamesOnly = !hasToolDetails && Boolean(debug.toolCalls && debug.toolCalls.length > 0);
  const hasToolsSection = debug.steps != null || hasToolDetails || hasToolNamesOnly;

  return (
    <div className="collapsible-panel collapsible-panel--debug">
      <button
        type="button"
        className="collapsible-panel__toggle"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
      >
        {expanded ? "▾" : "▸"} Debug
      </button>
      {expanded && (
        <div className="debug-panel">
          {debug.requestId && (
            <section className="debug-section" aria-label="Request">
              <h4 className="debug-section__title">Request</h4>
              <dl className="debug-grid">
                <dt>Request ID</dt>
                <dd>{debug.requestId}</dd>
              </dl>
            </section>
          )}

          {debug.route && (
            <section className="debug-section" aria-label="Pipeline">
              <h4 className="debug-section__title">Pipeline</h4>
              <dl className="debug-grid">
                <dt>Route</dt>
                <dd>{debug.route}</dd>
                {debug.terminationReason && (
                  <>
                    <dt>Termination reason</dt>
                    <dd>{debug.terminationReason}</dd>
                  </>
                )}
              </dl>
            </section>
          )}

          {(debug.retrievalMs != null || debug.generationMs != null || debug.totalMs != null) && (
            <section className="debug-section" aria-label="Timings">
              <h4 className="debug-section__title">Timings</h4>
              <dl className="debug-grid">
                {debug.retrievalMs != null && (
                  <>
                    <dt>Retrieval</dt>
                    <dd>{Math.round(debug.retrievalMs)} ms</dd>
                  </>
                )}
                {debug.generationMs != null && (
                  <>
                    <dt>Generation</dt>
                    <dd>{Math.round(debug.generationMs)} ms</dd>
                  </>
                )}
                {debug.totalMs != null && (
                  <>
                    <dt>Total</dt>
                    <dd>{Math.round(debug.totalMs)} ms</dd>
                  </>
                )}
              </dl>
            </section>
          )}

          {hasToolsSection && (
            <section className="debug-section" aria-label="Agent and tool execution">
              <h4 className="debug-section__title">Tools</h4>
              {debug.steps != null && (
                <dl className="debug-grid">
                  <dt>Steps</dt>
                  <dd>{debug.steps}</dd>
                </dl>
              )}
              {hasToolDetails && (
                <ul className="tool-call-list">
                  {debug.toolCallDetails!.map((detail, index) => (
                    <ToolCallCard key={`${detail.tool_name}-${index}`} detail={detail} />
                  ))}
                </ul>
              )}
              {hasToolNamesOnly && (
                <dl className="debug-grid">
                  <dt>Tool calls</dt>
                  <dd>{debug.toolCalls!.join(", ")}</dd>
                </dl>
              )}
            </section>
          )}
        </div>
      )}
    </div>
  );
}
