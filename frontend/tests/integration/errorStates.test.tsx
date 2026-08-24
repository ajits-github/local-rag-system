import userEvent from "@testing-library/user-event";
import { screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { jsonResponse, renderChatWindow, routedFetchMock } from "./testUtils";

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
});
