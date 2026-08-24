import userEvent from "@testing-library/user-event";
import { screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderChatWindow, routedFetchMock, sseResponse } from "./testUtils";

describe("malformed / disconnected stream behavior", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows a malformed-response error when the terminal frame doesn't match the expected shape", async () => {
    const badFrame = 'event: completed\ndata: {"not_a_valid_response": true}\n\n';
    vi.stubGlobal(
      "fetch",
      routedFetchMock({ "/agent/query/stream": () => sseResponse([badFrame]) })
    );

    renderChatWindow();
    const user = userEvent.setup();
    await user.click(screen.getByRole("radio", { name: "Agentic RAG" }));
    await user.type(screen.getByLabelText("Message"), "hi");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/could not understand/));
  });

  it("shows a malformed-response error when a stream frame contains unparsable JSON", async () => {
    const badFrame = "event: tool_started\ndata: {not-json\n\n";
    vi.stubGlobal(
      "fetch",
      routedFetchMock({ "/agent/query/stream": () => sseResponse([badFrame]) })
    );

    renderChatWindow();
    const user = userEvent.setup();
    await user.click(screen.getByRole("radio", { name: "Agentic RAG" }));
    await user.type(screen.getByLabelText("Message"), "hi");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/could not understand/));
  });

  it("surfaces a backend-unavailable error if the stream connection fails outright", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("network error")));

    renderChatWindow();
    const user = userEvent.setup();
    await user.click(screen.getByRole("radio", { name: "Agentic RAG" }));
    await user.type(screen.getByLabelText("Message"), "hi");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/Could not reach the backend/));
  });
});
