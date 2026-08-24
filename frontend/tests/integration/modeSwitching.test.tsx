import userEvent from "@testing-library/user-event";
import { screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { jsonResponse, renderChatWindow, routedFetchMock, sseFrame, sseResponse } from "./testUtils";

const agentCompletedFrame = sseFrame("completed", {
  answer: "a",
  sources: [],
  route: "agent",
  termination_reason: "synthesized",
  steps: 1,
  tool_calls: [],
  retrieval_ms: 1,
  generation_ms: 1,
  total_ms: 2,
});

describe("mode switching", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("defaults to Classic RAG and calls /query", async () => {
    const fetchMock = routedFetchMock({
      "/query": () =>
        jsonResponse(200, { answer: "classic answer", sources: [], retrieval_ms: 1, generation_ms: 1, total_ms: 2 }),
    });
    vi.stubGlobal("fetch", fetchMock);

    renderChatWindow();
    expect(screen.getByRole("radio", { name: "Classic RAG" })).toHaveAttribute("aria-checked", "true");

    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Message"), "hi");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/query", expect.anything()));
  });

  it("switches to Agentic RAG and calls the agent stream endpoint instead", async () => {
    const fetchMock = routedFetchMock({
      "/agent/query/stream": () => sseResponse([agentCompletedFrame]),
    });
    vi.stubGlobal("fetch", fetchMock);

    renderChatWindow();
    const user = userEvent.setup();
    await user.click(screen.getByRole("radio", { name: "Agentic RAG" }));

    expect(screen.getByRole("radio", { name: "Agentic RAG" })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByRole("radio", { name: "Classic RAG" })).toHaveAttribute("aria-checked", "false");

    await user.type(screen.getByLabelText("Message"), "hi");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/agent/query/stream", expect.anything()));
  });

  it("keeps separate mode selection per new chat (mode is not reset by New chat)", async () => {
    vi.stubGlobal(
      "fetch",
      routedFetchMock({
        "/agent/query/stream": () => sseResponse([agentCompletedFrame]),
      })
    );
    renderChatWindow();
    const user = userEvent.setup();
    await user.click(screen.getByRole("radio", { name: "Agentic RAG" }));
    await user.type(screen.getByLabelText("Message"), "hi");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "New chat" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "New chat" }));

    expect(screen.getByRole("radio", { name: "Agentic RAG" })).toHaveAttribute("aria-checked", "true");
  });
});
