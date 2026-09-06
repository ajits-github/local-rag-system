import { describe, expect, it } from "vitest";
import { buildQueryRequestBody } from "../../src/api/query";
import { buildAgentQueryRequestBody } from "../../src/api/agentQuery";
import type { DevIdentity } from "../../src/api/types";

const emptyIdentity: DevIdentity = {
  bearerToken: "",
  tenantId: "",
  roles: "",
  asOf: "",
  requireTrustLevel: "",
  datasetId: "",
};

describe("buildQueryRequestBody dataset_id handling", () => {
  it("omits filters entirely when no dataset is set (existing behavior unchanged)", () => {
    const body = buildQueryRequestBody("hello", emptyIdentity);
    expect(body.filters).toBeUndefined();
  });

  it("sets filters.dataset_id when a dataset is chosen", () => {
    const body = buildQueryRequestBody("hello", { ...emptyIdentity, datasetId: "techfusion" });
    expect(body.filters).toEqual({ dataset_id: "techfusion" });
  });

  it("still scopes retrieval even with a bearer token present", () => {
    // dataset_id is a retrieval filter, not an identity claim -- it must never
    // be suppressed by the same bearer-token precedence rule that hides
    // tenant_id/roles once a verified JWT identity exists.
    const body = buildQueryRequestBody("hello", {
      ...emptyIdentity,
      bearerToken: "some-jwt",
      datasetId: "techfusion",
    });
    expect(body.filters).toEqual({ dataset_id: "techfusion" });
  });
});

describe("buildAgentQueryRequestBody dataset_id handling", () => {
  it("omits filters entirely when no dataset is set (existing behavior unchanged)", () => {
    const body = buildAgentQueryRequestBody("hello", emptyIdentity);
    expect(body.filters).toBeUndefined();
  });

  it("sets filters.dataset_id when a dataset is chosen", () => {
    const body = buildAgentQueryRequestBody("hello", { ...emptyIdentity, datasetId: "techfusion" });
    expect(body.filters).toEqual({ dataset_id: "techfusion" });
  });
});
