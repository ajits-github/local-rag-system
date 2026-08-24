import { useState } from "react";
import type { DevIdentity } from "../api/types";

/**
 * Local-development-only identity controls. Never trusted client-side:
 * everything here is just what gets sent in the request. The backend
 * (rag.api.request_auth.build_authorization_context) ignores tenant_id/
 * roles entirely whenever a verified JWT identity is present, so this
 * panel disables those two fields the moment a bearer token is entered,
 * to avoid implying they still have any effect.
 */
export function DevIdentityPanel({
  identity,
  onChange,
}: {
  identity: DevIdentity;
  onChange: (identity: DevIdentity) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const hasToken = identity.bearerToken.trim().length > 0;

  const update = (patch: Partial<DevIdentity>) => onChange({ ...identity, ...patch });

  return (
    <div className="collapsible-panel collapsible-panel--dev">
      <button
        type="button"
        className="collapsible-panel__toggle"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
      >
        {expanded ? "▾" : "▸"} Developer settings
      </button>
      {expanded && (
        <div className="dev-identity-form">
          <p className="dev-identity-form__notice">
            Local development only. These values are sent with each request; the backend decides whether
            they are honored (see the project&apos;s authentication docs).
          </p>
          <label>
            Bearer token
            <input
              type="password"
              value={identity.bearerToken}
              onChange={(e) => update({ bearerToken: e.target.value })}
              placeholder="Paste a JWT for authenticated requests"
              autoComplete="off"
            />
          </label>
          <label className={hasToken ? "is-disabled" : ""}>
            Tenant ID
            <input
              type="text"
              value={identity.tenantId}
              onChange={(e) => update({ tenantId: e.target.value })}
              disabled={hasToken}
              placeholder={hasToken ? "Ignored: identity comes from the bearer token" : "e.g. tenant-alpha"}
            />
          </label>
          <label className={hasToken ? "is-disabled" : ""}>
            Roles (comma-separated)
            <input
              type="text"
              value={identity.roles}
              onChange={(e) => update({ roles: e.target.value })}
              disabled={hasToken}
              placeholder={hasToken ? "Ignored: identity comes from the bearer token" : "e.g. viewer, support"}
            />
          </label>
          {hasToken && (
            <p className="dev-identity-form__hint">
              Tenant/roles are ignored while a bearer token is set -- the verified JWT identity takes
              precedence.
            </p>
          )}
          <label>
            As of (date)
            <input type="date" value={identity.asOf} onChange={(e) => update({ asOf: e.target.value })} />
          </label>
          <label>
            Require trust level
            <input
              type="text"
              value={identity.requireTrustLevel}
              onChange={(e) => update({ requireTrustLevel: e.target.value })}
              placeholder="e.g. authoritative"
            />
          </label>
        </div>
      )}
    </div>
  );
}
