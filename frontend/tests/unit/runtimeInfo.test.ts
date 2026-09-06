import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchRuntimeInfo } from "../../src/api/runtimeInfo";
import { RagApiError } from "../../src/api/client";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

const validBody = {
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
};

describe("fetchRuntimeInfo", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns the parsed runtime info on a valid response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(200, validBody)));
    const info = await fetchRuntimeInfo();
    expect(info.generation_model).toBe("qwen2.5:1.5b");
    expect(info.retrieval_mode).toBe("dense");
    expect(info.fusion_method).toBeNull();
  });

  it("parses a hybrid-mode response with a fusion method set", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(200, { ...validBody, retrieval_mode: "hybrid", sparse_retrieval_enabled: true, fusion_method: "rrf" })
      )
    );
    const info = await fetchRuntimeInfo();
    expect(info.sparse_retrieval_enabled).toBe(true);
    expect(info.fusion_method).toBe("rrf");
  });

  it("throws malformed_response when the body doesn't match the schema", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(200, { generation_provider: "ollama" })));
    await expect(fetchRuntimeInfo()).rejects.toMatchObject({ kind: "malformed_response" } satisfies Partial<RagApiError>);
  });

  it("throws server_error on a non-2xx response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(500, { detail: "boom" })));
    await expect(fetchRuntimeInfo()).rejects.toMatchObject({ kind: "server_error" });
  });

  it("throws backend_unavailable on a network failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(fetchRuntimeInfo()).rejects.toMatchObject({ kind: "backend_unavailable" });
  });
});
