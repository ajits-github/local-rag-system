import { describe, expect, it } from "vitest";
import { decodeJwt, extractDisplayClaims, looksLikeJwt } from "../../src/utils/jwt";
import { makeJwt } from "../testHelpers";

describe("decodeJwt", () => {
  it("decodes a well-formed token's header and payload", () => {
    const token = makeJwt({ sub: "user-1", tenant_id: "tenant-alpha", roles: ["viewer"], exp: 9999999999 });
    const decoded = decodeJwt(token);
    expect(decoded).not.toBeNull();
    expect(decoded!.payload).toMatchObject({ sub: "user-1", tenant_id: "tenant-alpha", exp: 9999999999 });
    expect(decoded!.header).toMatchObject({ alg: "HS256", typ: "JWT" });
  });

  it("returns null for a string with the wrong number of segments", () => {
    expect(decodeJwt("not-a-jwt")).toBeNull();
    expect(decodeJwt("only.two")).toBeNull();
    expect(decodeJwt("a.b.c.d")).toBeNull();
  });

  it("returns null for a segment that isn't valid base64url JSON", () => {
    expect(decodeJwt("!!!.!!!.sig")).toBeNull();
  });

  it("returns null when a segment decodes to non-object JSON", () => {
    const notAnObject = btoa(JSON.stringify("just a string")).replace(/=+$/, "");
    expect(decodeJwt(`${notAnObject}.${notAnObject}.sig`)).toBeNull();
  });

  it("returns null for an empty-segment token", () => {
    expect(decodeJwt("..sig")).toBeNull();
  });
});

describe("looksLikeJwt", () => {
  it("is true for a structurally valid token", () => {
    expect(looksLikeJwt(makeJwt({ sub: "x" }))).toBe(true);
  });

  it("is false for arbitrary text", () => {
    expect(looksLikeJwt("signed.jwt.token")).toBe(false);
    expect(looksLikeJwt("")).toBe(false);
  });
});

describe("extractDisplayClaims", () => {
  it("extracts only the known display claims", () => {
    const claims = extractDisplayClaims({
      exp: 123,
      sub: "user-1",
      tenant_id: "tenant-alpha",
      roles: ["viewer", "support"],
      secret_internal_field: "should-not-leak",
    });
    expect(claims).toEqual({ exp: 123, sub: "user-1", tenantId: "tenant-alpha", roles: ["viewer", "support"] });
    expect(claims).not.toHaveProperty("secret_internal_field");
  });

  it("omits claims that are missing or the wrong type", () => {
    expect(extractDisplayClaims({ roles: "not-an-array" })).toEqual({});
    expect(extractDisplayClaims({})).toEqual({});
  });
});
