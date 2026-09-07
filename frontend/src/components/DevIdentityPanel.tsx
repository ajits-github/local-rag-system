import { Fragment, forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import type { DevIdentity } from "../api/types";
import { formatApproxDuration, useTokenStatus, type TokenStatus } from "../hooks/useTokenStatus";
import { looksLikeJwt } from "../utils/jwt";

/** Imperative handle so a sibling (the header's TokenStatusBadge "Replace token" action) can open this panel. */
export interface DevIdentityPanelHandle {
  openForEdit: () => void;
}

type FieldErrors = Partial<Record<keyof DevIdentity, string>>;

function hasAnyIdentityConfigured(identity: DevIdentity): boolean {
  return Object.values(identity).some((v) => v.trim().length > 0);
}

function validateDraft(draft: DevIdentity): FieldErrors {
  const errors: FieldErrors = {};
  const token = draft.bearerToken.trim();
  if (token && !looksLikeJwt(token)) {
    errors.bearerToken = "This doesn't look like a valid JWT (expected header.payload.signature).";
  }
  if (draft.asOf) {
    const parsed = new Date(draft.asOf);
    if (Number.isNaN(parsed.getTime())) {
      errors.asOf = "Enter a valid date.";
    }
  }
  return errors;
}

/**
 * One short segment summarizing token health, meant to sit inline among
 * the other facts rather than as its own paragraph -- the header's
 * TokenStatusBadge already owns the live "Authenticated" countdown, so
 * this only adds a word when it says something that badge doesn't
 * already cover (no-exp token: authenticated but no countdown to show;
 * expired/malformed: worth restating next to Edit settings).
 */
function statusSegment(status: TokenStatus): string | null {
  if (!status.hasToken) return null;
  switch (status.state) {
    case "invalid":
      return "Malformed token";
    case "expired":
      return "Token expired";
    case "no_exp":
      return "Authenticated";
    default:
      return status.remainingMs != null ? `expires in ${formatApproxDuration(status.remainingMs)}` : null;
  }
}

/**
 * Compact horizontal summary (e.g. "tenant_beta · tenant_beta_operator ·
 * Current · expires in 3h 58m") instead of a multi-row field list --
 * detailed editing is always one click away via Edit settings, so the
 * collapsed view doesn't need to spend a whole row per field.
 */
function DevSessionSummary({
  identity,
  onEdit,
  editButtonRef,
}: {
  identity: DevIdentity;
  onEdit: () => void;
  editButtonRef: React.RefObject<HTMLButtonElement>;
}) {
  const status = useTokenStatus(identity.bearerToken);
  const tenantLabel = status.hasToken ? status.claims?.tenantId : identity.tenantId || undefined;
  const roles = status.hasToken
    ? status.claims?.roles
    : identity.roles
        .split(",")
        .map((r) => r.trim())
        .filter(Boolean);
  const rolesLabel = roles && roles.length > 0 ? roles.join(", ") : undefined;

  const segments = [
    tenantLabel,
    rolesLabel,
    identity.asOf || "Current",
    identity.datasetId || undefined,
    identity.requireTrustLevel || undefined,
    statusSegment(status) ?? undefined,
  ].filter((segment): segment is string => Boolean(segment));

  return (
    <div className="dev-session-summary">
      <div className="dev-session-summary__header">
        <span className="dev-session-summary__title">Developer session</span>
        <button type="button" ref={editButtonRef} className="dev-session-summary__edit" onClick={onEdit}>
          Edit settings
        </button>
      </div>
      <p className="dev-session-summary__line">
        {segments.map((segment, index) => (
          <Fragment key={index}>
            {index > 0 && (
              <span className="dev-session-summary__sep" aria-hidden="true">
                ·
              </span>
            )}
            <span className="dev-session-summary__field">{segment}</span>
          </Fragment>
        ))}
      </p>
    </div>
  );
}

/**
 * Local-development-only identity controls. Never trusted client-side:
 * everything here is just what gets sent in the request. The backend
 * (rag.api.request_auth.build_authorization_context) ignores tenant_id/
 * roles entirely whenever a verified JWT identity is present, so this
 * panel disables those two fields the moment a bearer token is entered,
 * to avoid implying they still have any effect.
 *
 * Fields are edited in a local draft and only take effect on "Apply
 * settings" (validated first) -- previously every keystroke immediately
 * changed what the next request would send, with no clear moment where a
 * caller could tell "this is the configuration currently in use."
 */
export const DevIdentityPanel = forwardRef<
  DevIdentityPanelHandle,
  { identity: DevIdentity; onChange: (identity: DevIdentity) => void }
>(function DevIdentityPanel({ identity, onChange }, ref) {
  const [expanded, setExpanded] = useState(false);
  const [draft, setDraft] = useState<DevIdentity>(identity);
  const [errors, setErrors] = useState<FieldErrors>({});
  const bearerTokenRef = useRef<HTMLInputElement>(null);
  const asOfRef = useRef<HTMLInputElement>(null);
  const editButtonRef = useRef<HTMLButtonElement>(null);
  const focusEditButtonOnNextRender = useRef(false);

  useEffect(() => {
    if (focusEditButtonOnNextRender.current) {
      focusEditButtonOnNextRender.current = false;
      editButtonRef.current?.focus();
    }
  });

  useEffect(() => {
    if (expanded) bearerTokenRef.current?.focus();
  }, [expanded]);

  const hasApplied = hasAnyIdentityConfigured(identity);
  const hasToken = draft.bearerToken.trim().length > 0;

  const update = (patch: Partial<DevIdentity>) => setDraft((d) => ({ ...d, ...patch }));

  function openForEdit() {
    setDraft(identity);
    setErrors({});
    setExpanded(true);
  }

  useImperativeHandle(ref, () => ({ openForEdit }));

  function collapseWithoutApplying() {
    setExpanded(false);
  }

  function handleApply(e: React.FormEvent) {
    e.preventDefault();
    const validationErrors = validateDraft(draft);
    setErrors(validationErrors);
    if (Object.keys(validationErrors).length > 0) {
      if (validationErrors.bearerToken) bearerTokenRef.current?.focus();
      else if (validationErrors.asOf) asOfRef.current?.focus();
      return;
    }
    onChange(draft);
    focusEditButtonOnNextRender.current = true;
    setExpanded(false);
  }

  if (!expanded && hasApplied) {
    return <DevSessionSummary identity={identity} onEdit={openForEdit} editButtonRef={editButtonRef} />;
  }

  if (!expanded) {
    return (
      <div className="collapsible-panel collapsible-panel--dev">
        <button
          type="button"
          className="collapsible-panel__toggle"
          onClick={openForEdit}
          aria-expanded={false}
        >
          ▸ Developer settings
        </button>
      </div>
    );
  }

  return (
    <div className="collapsible-panel collapsible-panel--dev">
      <button
        type="button"
        className="collapsible-panel__toggle"
        onClick={collapseWithoutApplying}
        aria-expanded={true}
      >
        ▾ Developer settings
      </button>
      <form className="dev-identity-form" onSubmit={handleApply} noValidate>
        <p className="dev-identity-form__notice">
          Local development only. These values take effect once you apply them; the backend decides
          whether they are honored (see the project&apos;s authentication docs).
        </p>
        <label>
          Bearer token
          <input
            ref={bearerTokenRef}
            type="password"
            value={draft.bearerToken}
            onChange={(e) => update({ bearerToken: e.target.value })}
            placeholder="Paste a JWT for authenticated requests"
            autoComplete="off"
            aria-invalid={errors.bearerToken ? true : undefined}
            aria-describedby={errors.bearerToken ? "dev-identity-bearer-error" : undefined}
          />
        </label>
        {errors.bearerToken && (
          <p id="dev-identity-bearer-error" className="dev-identity-form__error" role="alert">
            {errors.bearerToken}
          </p>
        )}
        <label className={hasToken ? "is-disabled" : ""}>
          Tenant ID
          <input
            type="text"
            value={draft.tenantId}
            onChange={(e) => update({ tenantId: e.target.value })}
            disabled={hasToken}
            placeholder={hasToken ? "Ignored: identity comes from the bearer token" : "e.g. tenant-alpha"}
          />
        </label>
        <label className={hasToken ? "is-disabled" : ""}>
          Roles (comma-separated)
          <input
            type="text"
            value={draft.roles}
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
          <input
            ref={asOfRef}
            type="date"
            value={draft.asOf}
            onChange={(e) => update({ asOf: e.target.value })}
            aria-invalid={errors.asOf ? true : undefined}
            aria-describedby={errors.asOf ? "dev-identity-asof-error" : undefined}
          />
        </label>
        {errors.asOf && (
          <p id="dev-identity-asof-error" className="dev-identity-form__error" role="alert">
            {errors.asOf}
          </p>
        )}
        <label>
          Require trust level
          <input
            type="text"
            value={draft.requireTrustLevel}
            onChange={(e) => update({ requireTrustLevel: e.target.value })}
            placeholder="e.g. authoritative"
          />
        </label>
        <label>
          Dataset ID
          <input
            type="text"
            value={draft.datasetId}
            onChange={(e) => update({ datasetId: e.target.value })}
            placeholder="e.g. techfusion (leave blank to search every ingested dataset)"
          />
        </label>
        {!draft.datasetId && (
          <p className="dev-identity-form__hint">
            No dataset selected: retrieval will search every dataset_id ever ingested into this
            Postgres instance, not just one corpus. Set this to match the corpus you actually mean to
            query.
          </p>
        )}
        <div className="dev-identity-form__actions">
          <button type="submit" className="dev-identity-form__apply">
            Apply settings
          </button>
        </div>
      </form>
    </div>
  );
});
