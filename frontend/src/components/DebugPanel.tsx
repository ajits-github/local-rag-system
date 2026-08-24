import { useState } from "react";
import type { DebugInfo } from "../state/types";

export function DebugPanel({ debug }: { debug: DebugInfo | undefined }) {
  const [expanded, setExpanded] = useState(false);
  if (!debug) return null;

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
        <dl className="debug-grid">
          {debug.route && (
            <>
              <dt>Route</dt>
              <dd>{debug.route}</dd>
            </>
          )}
          {debug.steps != null && (
            <>
              <dt>Steps</dt>
              <dd>{debug.steps}</dd>
            </>
          )}
          {debug.toolCalls && debug.toolCalls.length > 0 && (
            <>
              <dt>Tool calls</dt>
              <dd>{debug.toolCalls.join(", ")}</dd>
            </>
          )}
          {debug.terminationReason && (
            <>
              <dt>Termination reason</dt>
              <dd>{debug.terminationReason}</dd>
            </>
          )}
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
      )}
    </div>
  );
}
