import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AgentActivityPanel } from "../../src/components/AgentActivityPanel";
import type { AgentEvent } from "../../src/api/types";

describe("AgentActivityPanel", () => {
  it("renders known event fields", () => {
    const events: AgentEvent[] = [
      { event_type: "tool_started", tool_name: "search_knowledge_base", step: 2 },
      { event_type: "evidence_evaluated", evidence_sufficient: true },
    ];
    render(<AgentActivityPanel events={events} isLive={false} />);

    expect(screen.getByText(/Tool executing/)).toBeInTheDocument();
    expect(screen.getByText(/search_knowledge_base/)).toBeInTheDocument();
    expect(screen.getByText(/Evidence evaluated/)).toBeInTheDocument();
  });

  it("never renders a hidden reasoning/free-text field, even if a malformed event carries one", () => {
    // Simulate a buggy/compromised backend attaching an extra field the
    // AgentEvent schema doesn't define. AgentEvent has no free-text field
    // by construction (see rag/agent/events.py); this proves the UI layer
    // doesn't accidentally surface one even if raw JSON did carry it.
    const suspicious = {
      event_type: "tool_started",
      tool_name: "search_knowledge_base",
      reasoning: "SECRET_CHAIN_OF_THOUGHT_TEXT",
      raw_prompt: "SECRET_PROMPT_TEXT",
    } as unknown as AgentEvent;

    render(<AgentActivityPanel events={[suspicious]} isLive={false} />);

    expect(screen.queryByText(/SECRET_CHAIN_OF_THOUGHT_TEXT/)).not.toBeInTheDocument();
    expect(screen.queryByText(/SECRET_PROMPT_TEXT/)).not.toBeInTheDocument();
  });

  it("shows a notice when the live stream fell back to the non-streaming endpoint", () => {
    render(<AgentActivityPanel events={[]} isLive={false} streamFellBack />);
    expect(screen.getByText(/Live progress is disabled/)).toBeInTheDocument();
  });

  it("renders nothing when there are no events and no fallback notice", () => {
    const { container } = render(<AgentActivityPanel events={[]} isLive={false} />);
    expect(container).toBeEmptyDOMElement();
  });
});
