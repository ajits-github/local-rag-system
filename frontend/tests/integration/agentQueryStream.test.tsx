import userEvent from "@testing-library/user-event";
import { screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { jsonResponse, renderChatWindow, routedFetchMock, sseFrame, sseResponse } from "./testUtils";

describe("sending an agent query", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("streams live agent events and then renders the final answer", async () => {
    const frames =
      sseFrame("route_selected", { event_type: "route_selected", route: "agent" }) +
      sseFrame("tool_selected", { event_type: "tool_selected", tool_name: "search_knowledge_base", step: 1 }) +
      sseFrame("tool_completed", {
        event_type: "tool_completed",
        tool_name: "search_knowledge_base",
        retrieved_chunk_count: 4,
        step: 1,
      }) +
      sseFrame("completed", {
        answer: "Multi-hop answer synthesized from two documents.",
        sources: [
          {
            chunk_id: "c1",
            document_id: "d1",
            source: "knowledge_base/architecture/overview.md",
            score: 0.8,
          },
        ],
        route: "agent",
        termination_reason: "synthesized",
        steps: 3,
        tool_calls: ["search_knowledge_base"],
        retrieval_ms: 40,
        generation_ms: 900,
        total_ms: 950,
      });

    const fetchMock = routedFetchMock({
      "/agent/query/stream": () => sseResponse([frames]),
    });
    vi.stubGlobal("fetch", fetchMock);

    renderChatWindow();
    const user = userEvent.setup();
    await user.click(screen.getByRole("radio", { name: "Agentic RAG" }));
    await user.type(screen.getByLabelText("Message"), "How do ingestion and retrieval fit together?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(screen.getByText(/Tool executing|Tool completed/)).toBeInTheDocument());
    expect(screen.getAllByText(/search_knowledge_base/).length).toBeGreaterThan(0);

    await waitFor(() => expect(screen.getByText(/Multi-hop answer synthesized/)).toBeInTheDocument());

    expect(fetchMock.mock.calls.some(([url]) => url === "/agent/query/stream")).toBe(true);
  });

  it("falls back to the non-streaming endpoint when live events are disabled (404)", async () => {
    const fetchMock = routedFetchMock({
      "/agent/query/stream": () => jsonResponse(404, { detail: "Live agent events are disabled" }),
      "/agent/query": () =>
        jsonResponse(200, {
          answer: "Non-streamed agent answer.",
          sources: [],
          route: "classic_rag",
          termination_reason: "synthesized",
          steps: 0,
          tool_calls: [],
          retrieval_ms: 10,
          generation_ms: 200,
          total_ms: 210,
        }),
    });
    vi.stubGlobal("fetch", fetchMock);

    renderChatWindow();
    const user = userEvent.setup();
    await user.click(screen.getByRole("radio", { name: "Agentic RAG" }));
    await user.type(screen.getByLabelText("Message"), "Simple question");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(screen.getByText(/Non-streamed agent answer/)).toBeInTheDocument());
    expect(screen.getByText(/Live progress is disabled/)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/agent/query/stream", expect.anything());
    expect(fetchMock).toHaveBeenCalledWith("/agent/query", expect.anything());
  });
});
