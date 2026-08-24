import { afterEach, describe, expect, it, vi } from "vitest";
import { postJson, RagApiError, buildHeaders } from "../../src/api/client";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

describe("buildHeaders", () => {
  it("attaches an Authorization header only when a bearer token is present", () => {
    expect(buildHeaders(undefined)).not.toHaveProperty("Authorization");
    expect(buildHeaders({ bearerToken: "", tenantId: "", roles: "", asOf: "", requireTrustLevel: "" })).not.toHaveProperty(
      "Authorization"
    );
    expect(
      buildHeaders({ bearerToken: "abc.def.ghi", tenantId: "", roles: "", asOf: "", requireTrustLevel: "" })
    ).toMatchObject({ Authorization: "Bearer abc.def.ghi" });
  });
});

describe("postJson error classification", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("classifies 401 as authentication_failed", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(401, { detail: "missing_token" })));
    await expect(postJson("/query", { query: "x" })).rejects.toMatchObject({
      kind: "authentication_failed",
    } satisfies Partial<RagApiError>);
  });

  it("classifies 429 as rate_limited", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse(429, { detail: "Rate limit exceeded: 60 per 1 minute" }))
    );
    await expect(postJson("/query", { query: "x" })).rejects.toMatchObject({ kind: "rate_limited" });
  });

  it("classifies 422 as validation_failed and extracts a FastAPI-list-shaped detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(422, { detail: [{ loc: ["body", "query"], msg: "field required", type: "value_error" }] })
      )
    );
    const err = await postJson("/query", {}).catch((e) => e);
    expect(err).toBeInstanceOf(RagApiError);
    expect(err.kind).toBe("validation_failed");
    expect(err.detail).toContain("field required");
  });

  it("classifies a network failure as backend_unavailable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(postJson("/query", { query: "x" })).rejects.toMatchObject({ kind: "backend_unavailable" });
  });

  it("resolves with the raw response on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(200, { answer: "hi", sources: [] })));
    const response = await postJson("/query", { query: "x" });
    expect(response.ok).toBe(true);
  });
});
