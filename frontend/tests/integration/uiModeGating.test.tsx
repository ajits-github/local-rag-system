import userEvent from "@testing-library/user-event";
import { screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { expInMinutes, makeJwt } from "../testHelpers";
import { jsonResponse, renderChatWindow, routedFetchMock } from "./testUtils";

/**
 * Proves the developer-vs-production UI mode gating (src/config/uiMode.ts)
 * end to end through the real ChatWindow, mirroring how a build's
 * VITE_UI_MODE actually reaches these components. Component-level
 * gating details (e.g. SourcesPanel's showScore prop) have their own
 * dedicated unit tests; this file is about what a caller actually sees.
 */
function queryResponseWithSource() {
  return jsonResponse(200, {
    answer: "The password policy requires a 12-character minimum.",
    sources: [
      {
        chunk_id: "c1",
        document_id: "d1",
        source: "knowledge_base/security/password-policy.md",
        category: "security",
        score: 0.913,
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
    request_id: "req-1",
  });
}

describe("developer vs. production UI mode", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("production mode (VITE_UI_MODE unset, the safe default) hides developer controls but still supports normal chat", async () => {
    vi.stubGlobal("fetch", routedFetchMock({ "/query": () => queryResponseWithSource() }));
    renderChatWindow();
    const user = userEvent.setup();

    expect(screen.queryByRole("button", { name: /Developer settings/ })).not.toBeInTheDocument();
    expect(screen.queryByText("Developer Mode")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Refresh backend status" })).not.toBeInTheDocument();

    await user.type(screen.getByLabelText("Message"), "What is the password policy?");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(screen.getByText(/12-character minimum/)).toBeInTheDocument());

    // Chat itself, and Classic/Agentic mode switching, keep working.
    expect(screen.getByRole("radio", { name: "Classic RAG" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Agentic RAG" })).toBeInTheDocument();

    // Detailed internal debug/tool information is hidden entirely.
    expect(screen.queryByRole("button", { name: /Debug/ })).not.toBeInTheDocument();

    // Citations/sources stay -- a keep item -- but the raw retrieval score is hidden.
    await user.click(screen.getByRole("button", { name: /Sources \(1\)/ }));
    expect(screen.getByText("knowledge_base/security/password-policy.md")).toBeInTheDocument();
    expect(screen.getByText("Requirements")).toBeInTheDocument();
    expect(screen.queryByText("0.913")).not.toBeInTheDocument();
  });

  it("a malformed VITE_UI_MODE value falls back to the same safe production behavior", async () => {
    vi.stubEnv("VITE_UI_MODE", "Developer"); // wrong case: must not enable developer mode
    vi.stubGlobal("fetch", routedFetchMock({}));
    renderChatWindow();

    expect(screen.queryByRole("button", { name: /Developer settings/ })).not.toBeInTheDocument();
    expect(screen.queryByText("Developer Mode")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Refresh backend status" })).not.toBeInTheDocument();
  });

  it("developer mode shows developer controls, runtime details, and debug/tool details", async () => {
    vi.stubEnv("VITE_UI_MODE", "developer");
    vi.stubGlobal("fetch", routedFetchMock({ "/query": () => queryResponseWithSource() }));
    renderChatWindow();
    const user = userEvent.setup();

    expect(screen.getByText("Developer Mode")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Developer settings/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Refresh backend status" })).toBeInTheDocument();

    await user.type(screen.getByLabelText("Message"), "What is the password policy?");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(screen.getByText(/12-character minimum/)).toBeInTheDocument());

    // The per-answer Debug panel is available in developer mode.
    await user.click(screen.getByRole("button", { name: /Debug/ }));
    expect(screen.getByText("req-1")).toBeInTheDocument();

    // The raw retrieval score is visible in developer mode.
    await user.click(screen.getByRole("button", { name: /Sources \(1\)/ }));
    expect(screen.getByText("0.913")).toBeInTheDocument();
  });

  it("developer mode still supports the JWT expiry countdown/session summary", async () => {
    vi.stubEnv("VITE_UI_MODE", "developer");
    vi.stubGlobal("fetch", routedFetchMock({}));
    renderChatWindow();
    const user = userEvent.setup();
    const token = makeJwt({ sub: "dev-user", tenant_id: "tenant-alpha", exp: expInMinutes(60) });

    await user.click(screen.getByRole("button", { name: /Developer settings/ }));
    await user.type(screen.getByLabelText("Bearer token"), token);
    await user.click(screen.getByRole("button", { name: "Apply settings" }));

    await waitFor(() => expect(screen.getByText("Developer session")).toBeInTheDocument());
    // Both the session summary and the header's TokenStatusBadge say so.
    expect(screen.getAllByText(/Authenticated/).length).toBeGreaterThan(0);
  });

  it("hiding Developer settings in production never changes what is actually sent to the backend", async () => {
    const fetchMock = routedFetchMock({ "/query": () => queryResponseWithSource() });
    vi.stubGlobal("fetch", fetchMock);
    renderChatWindow();
    const user = userEvent.setup();

    await user.type(screen.getByLabelText("Message"), "hi");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => url === "/query")).toBe(true));

    const [, init] = fetchMock.mock.calls.find(([url]) => url === "/query")!;
    const headers = init!.headers as Record<string, string>;
    expect(headers.Authorization).toBeUndefined();
    expect(JSON.parse(init!.body as string)).not.toHaveProperty("tenant_id");
    expect(JSON.parse(init!.body as string)).not.toHaveProperty("roles");
  });
});
