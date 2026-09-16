import userEvent from "@testing-library/user-event";
import { screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  jsonResponse,
  renderChatWindow,
  rootInfoResponse,
  routedFetchMock,
  runtimeInfoResponse,
} from "./testUtils";

async function sendMessage(text = "hello") {
  const user = userEvent.setup();
  await user.type(screen.getByLabelText("Message"), text);
  await user.click(screen.getByRole("button", { name: "Send" }));
}

describe("error states", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows an authentication-failure banner on 401, not a generic answer", async () => {
    vi.stubGlobal(
      "fetch",
      routedFetchMock({ "/query": () => jsonResponse(401, { detail: "missing_token" }) })
    );
    renderChatWindow();
    await sendMessage();
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/Authentication failed/));
  });

  it("shows a rate-limit banner on 429", async () => {
    vi.stubGlobal(
      "fetch",
      routedFetchMock({
        "/query": () => jsonResponse(429, { detail: "Rate limit exceeded: 60 per 1 minute" }),
      })
    );
    renderChatWindow();
    await sendMessage();
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/Rate limit exceeded/));
  });

  it("shows an oversized-request banner on 422", async () => {
    vi.stubGlobal(
      "fetch",
      routedFetchMock({
        "/query": () => jsonResponse(422, { detail: "query exceeds maximum length of 2000 characters" }),
      })
    );
    renderChatWindow();
    await sendMessage();
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/rejected as invalid/));
  });

  it("shows a backend-unavailable banner on a network failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    renderChatWindow();
    await sendMessage();
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/Could not reach the backend/));
  });

  it("shows a malformed-response banner when the server response doesn't match the expected shape", async () => {
    vi.stubGlobal(
      "fetch",
      routedFetchMock({ "/query": () => jsonResponse(200, { unexpected: "shape" }) })
    );
    renderChatWindow();
    await sendMessage();
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/could not understand/));
  });

  it("cancels an in-flight request via the Stop button and shows a neutral cancelled notice, not an error", async () => {
    // Simulates a real aborted fetch(): the /query call never resolves on
    // its own, only rejecting once the request's AbortSignal fires (which
    // clicking Stop triggers via useSendMessage's cancel()).
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string, init?: RequestInit) => {
        if (url === "/") return Promise.resolve(rootInfoResponse());
        if (url === "/info") return Promise.resolve(runtimeInfoResponse());
        if (url === "/query") {
          return new Promise((_resolve, reject) => {
            init?.signal?.addEventListener("abort", () => {
              const abortError = new Error("The operation was aborted.");
              abortError.name = "AbortError";
              reject(abortError);
            });
          });
        }
        throw new Error(`Unmocked fetch call to ${url}`);
      })
    );
    renderChatWindow();
    await sendMessage();

    const stopButton = await screen.findByRole("button", { name: "Stop" });
    const user = userEvent.setup();
    await user.click(stopButton);

    await waitFor(() => expect(screen.getByText("Request cancelled.")).toBeInTheDocument());
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    // The composer returns to its normal Send state, ready for another message.
    expect(screen.getByRole("button", { name: "Send" })).toBeInTheDocument();
  });
});
