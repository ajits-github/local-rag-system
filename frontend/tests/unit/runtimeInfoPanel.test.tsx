import userEvent from "@testing-library/user-event";
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RuntimeInfoPanel } from "../../src/components/RuntimeInfoPanel";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function runtimeInfoBody(overrides: Record<string, unknown> = {}) {
  return {
    generation_provider: "ollama",
    generation_model: "qwen2.5:1.5b",
    embedding_provider: "sentence_transformers",
    embedding_model: "sentence-transformers/all-MiniLM-L6-v2",
    vectorstore_provider: "pgvector",
    retrieval_mode: "dense",
    sparse_retrieval_enabled: false,
    fusion_method: null,
    reranker_provider: "none",
    reranker_enabled: false,
    agent_enabled: false,
    mcp_enabled: false,
    mcp_client_enabled: false,
    vision_provider: "none",
    tracing_enabled: false,
    rate_limit_enabled: false,
    field_redaction_enabled: false,
    authorization_enabled: false,
    auth_enabled: false,
    ...overrides,
  };
}

describe("RuntimeInfoPanel", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("fetches on mount so the compact badge is populated without expanding", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, runtimeInfoBody()));
    vi.stubGlobal("fetch", fetchMock);

    render(<RuntimeInfoPanel />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(screen.getByRole("button", { name: /qwen2\.5:1\.5b/ })).toBeInTheDocument();
  });

  it("shows a compact model/retrieval/fusion badge in the collapsed toggle", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(
          200,
          runtimeInfoBody({ retrieval_mode: "hybrid", sparse_retrieval_enabled: true, fusion_method: "rrf", mcp_client_enabled: true })
        )
      )
    );

    render(<RuntimeInfoPanel />);

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /qwen2\.5:1\.5b · Hybrid · RRF · MCP/ })).toBeInTheDocument()
    );
  });

  it("does not show the detailed field breakdown until expanded", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse(200, runtimeInfoBody({ retrieval_mode: "hybrid", fusion_method: "rrf" })))
    );

    const user = userEvent.setup();
    render(<RuntimeInfoPanel />);
    await waitFor(() => expect(screen.getByRole("button", { name: /Runtime:/ })).toBeInTheDocument());

    expect(screen.queryByText("Fusion method")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Runtime:/ }));

    expect(screen.getByText("Fusion method")).toBeInTheDocument();
  });

  it("shows an error state when the backend is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    render(<RuntimeInfoPanel />);

    await waitFor(() => expect(screen.getByText("(unavailable)")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: /Runtime configuration/ })).toBeInTheDocument();
  });

  it("re-fetches when the refresh button is clicked", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(200, runtimeInfoBody({ agent_enabled: false })))
      .mockResolvedValueOnce(jsonResponse(200, runtimeInfoBody({ agent_enabled: true })));
    vi.stubGlobal("fetch", fetchMock);

    const user = userEvent.setup();
    render(<RuntimeInfoPanel />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    await user.click(screen.getByRole("button", { name: /Runtime:/ }));
    await user.click(screen.getByRole("button", { name: "Refresh" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
  });
});
