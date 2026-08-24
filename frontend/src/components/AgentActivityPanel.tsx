import { useState } from "react";
import type { AgentEvent } from "../api/types";

const EVENT_LABEL: Record<AgentEvent["event_type"], string> = {
  query_received: "Query received",
  route_selected: "Route selected",
  decomposition_started: "Decomposing query",
  decomposition_completed: "Decomposition completed",
  tool_selected: "Tool selected",
  tool_started: "Tool executing",
  tool_completed: "Tool completed",
  evidence_evaluated: "Evidence evaluated",
  retry_started: "Retrying",
  synthesis_started: "Synthesizing answer",
  completed: "Completed",
  terminated: "Terminated",
};

function EventRow({ event, index }: { event: AgentEvent; index: number }) {
  const details: string[] = [];
  if (event.step != null) details.push(`step ${event.step}`);
  if (event.tool_name) details.push(event.tool_name);
  if (event.retrieved_chunk_count != null) details.push(`${event.retrieved_chunk_count} chunks`);
  if (event.evidence_sufficient != null) {
    details.push(event.evidence_sufficient ? "evidence sufficient" : "evidence insufficient");
  }
  if (event.route) details.push(`route: ${event.route}`);
  if (event.termination_reason) details.push(event.termination_reason);
  if (event.elapsed_ms != null) details.push(`${Math.round(event.elapsed_ms)}ms`);

  return (
    <li className="agent-event-row" data-event-type={event.event_type}>
      <span className="agent-event-row__index">{index + 1}</span>
      <span className="agent-event-row__label">{EVENT_LABEL[event.event_type] ?? event.event_type}</span>
      {details.length > 0 && <span className="agent-event-row__details">{details.join(" · ")}</span>}
    </li>
  );
}

export function AgentActivityPanel({
  events,
  isLive,
  streamFellBack,
}: {
  events: AgentEvent[];
  isLive: boolean;
  streamFellBack?: boolean;
}) {
  const [expanded, setExpanded] = useState(true);
  if (events.length === 0 && !streamFellBack) return null;

  return (
    <div className="collapsible-panel collapsible-panel--activity">
      <button
        type="button"
        className="collapsible-panel__toggle"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
      >
        {expanded ? "▾" : "▸"} Agent activity {isLive && <span className="live-dot" aria-label="live" />}
      </button>
      {expanded && (
        <>
          {streamFellBack && (
            <p className="agent-activity__notice">
              Live progress is disabled on this backend; showing the final result only.
            </p>
          )}
          <ol className="agent-event-list">
            {events.map((event, index) => (
              <EventRow key={index} event={event} index={index} />
            ))}
          </ol>
        </>
      )}
    </div>
  );
}
