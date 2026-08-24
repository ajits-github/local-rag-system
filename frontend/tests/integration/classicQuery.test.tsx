import userEvent from "@testing-library/user-event";
import { screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { jsonResponse, renderChatWindow, routedFetchMock } from "./testUtils";

describe("sending a classic query", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("posts to /query and renders the answer with its sources", async () => {
    const fetchMock = routedFetchMock({
      "/query": () =>
        jsonResponse(200, {
          answer: "The **password policy** requires a 12-character minimum.",
          sources: [
            {
              chunk_id: "c1",
              document_id: "d1",
              source: "knowledge_base/security/password-policy.md",
              category: "security",
              score: 0.91,
              content_type: "prose",
              section_path: "Requirements",
              page: null,
              attachment_name: null,
              source_anchor: null,
              vision_generated: false,
            },
          ],
          retrieval_ms: 12.5,
          generation_ms: 340.2,
          total_ms: 352.7,
        }),
    });
    vi.stubGlobal("fetch", fetchMock);

    renderChatWindow();
    const user = userEvent.setup();

    await user.type(screen.getByLabelText("Message"), "What is the password policy?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(screen.getByText(/12-character minimum/)).toBeInTheDocument());

    const queryCall = fetchMock.mock.calls.find(([url]) => url === "/query");
    expect(queryCall).toBeDefined();
    const [, init] = queryCall!;
    expect(JSON.parse(init!.body as string)).toMatchObject({ query: "What is the password policy?" });

    // Sources panel is collapsed by default; expand it and check content.
    await user.click(screen.getByRole("button", { name: /Sources \(1\)/ }));
    expect(screen.getByText("knowledge_base/security/password-policy.md")).toBeInTheDocument();
    expect(screen.getByText("Requirements")).toBeInTheDocument();
  });

  it("shows an insufficient-evidence notice when no sources are returned", async () => {
    vi.stubGlobal(
      "fetch",
      routedFetchMock({
        "/query": () =>
          jsonResponse(200, {
            answer: "I don't have enough information to answer that.",
            sources: [],
            retrieval_ms: 5,
            generation_ms: 100,
            total_ms: 105,
          }),
      })
    );

    renderChatWindow();
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Message"), "Something obscure");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(screen.getByText(/Insufficient evidence/)).toBeInTheDocument());
  });
});
