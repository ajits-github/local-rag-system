import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchRootInfo } from "../../src/api/rootInfo";
import { RagApiError } from "../../src/api/client";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

const validBody = {
  service: "local-rag-system",
  status: "ok",
  docs: "/docs",
  health: "/health",
  metrics: "/metrics",
  features: {
    auth_enabled: false,
    insecure_dev_mode: false,
    authorization_enabled: true,
    field_redaction_enabled: true,
    rate_limit_enabled: false,
    agent_enabled: false,
    vision_provider: "none",
    tracing_enabled: false,
  },
};

describe("fetchRootInfo", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns the parsed features on a valid response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(200, validBody)));
    const info = await fetchRootInfo();
    expect(info.features.authorization_enabled).toBe(true);
    expect(info.features.field_redaction_enabled).toBe(true);
    expect(info.features.vision_provider).toBe("none");
  });

  it("throws malformed_response when the features block doesn't match the schema", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse(200, { ...validBody, features: { auth_enabled: "yes" } }))
    );
    await expect(fetchRootInfo()).rejects.toMatchObject({ kind: "malformed_response" } satisfies Partial<RagApiError>);
  });

  it("throws server_error on a non-2xx response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(500, { detail: "boom" })));
    await expect(fetchRootInfo()).rejects.toMatchObject({ kind: "server_error" });
  });

  it("throws backend_unavailable on a network failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(fetchRootInfo()).rejects.toMatchObject({ kind: "backend_unavailable" });
  });
});
