import userEvent from "@testing-library/user-event";
import { screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { jsonResponse, renderChatWindow, routedFetchMock } from "./testUtils";

describe("empty-state example queries", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("clicking an example chip populates the composer but does not send a request", async () => {
    const fetchMock = routedFetchMock({});
    vi.stubGlobal("fetch", fetchMock);
    renderChatWindow();
    const user = userEvent.setup();

    const chip = screen.getByRole("button", { name: "What is the webhook delivery timeout?" });
    await user.click(chip);

    expect(screen.getByLabelText("Message")).toHaveValue("What is the webhook delivery timeout?");
    expect(fetchMock.mock.calls.some(([url]) => url === "/query")).toBe(false);
  });

  it("the populated text can then be sent normally, same as manually typed text", async () => {
    const fetchMock = routedFetchMock({
      "/query": () => jsonResponse(200, { answer: "5 seconds.", sources: [], retrieval_ms: 1, generation_ms: 1, total_ms: 2 }),
    });
    vi.stubGlobal("fetch", fetchMock);
    renderChatWindow();
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: "What is the webhook delivery timeout?" }));
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => url === "/query")).toBe(true));
    const [, init] = fetchMock.mock.calls.find(([url]) => url === "/query")!;
    expect(JSON.parse(init!.body as string)).toMatchObject({ query: "What is the webhook delivery timeout?" });
  });
});
