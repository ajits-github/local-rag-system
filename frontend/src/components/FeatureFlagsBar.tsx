import { useBackendFeatures } from "../hooks/useBackendFeatures";
import type { FeatureFlags } from "../api/types";

const BOOLEAN_FLAGS: { key: keyof FeatureFlags & string; label: string }[] = [
  { key: "auth_enabled", label: "Auth" },
  { key: "authorization_enabled", label: "Authorization" },
  { key: "field_redaction_enabled", label: "Redaction" },
  { key: "rate_limit_enabled", label: "Rate limit" },
  { key: "agent_enabled", label: "Agent" },
  { key: "tracing_enabled", label: "Tracing" },
];

/**
 * Always-visible, at-a-glance summary of which optional security/agent
 * features the connected backend currently has active. Fetched from
 * GET / (rag.api.main.FeatureFlags), never guessed or cached from a prior
 * session. Exists so an insecure demo config (auth/authorization/field
 * redaction off) is never mistaken for a secure one just because the UI
 * looks the same either way.
 */
export function FeatureFlagsBar() {
  const { features, loading, error, refresh } = useBackendFeatures();

  return (
    <div className="feature-flags-bar" aria-label="Active backend features">
      {loading && <span className="feature-flags-bar__status">Checking backend configuration…</span>}
      {error && !loading && (
        <span className="feature-flags-bar__status feature-flags-bar__status--error" role="status">
          Backend status unavailable
        </span>
      )}
      {features && !loading && !error && (
        <>
          {BOOLEAN_FLAGS.map(({ key, label }) => {
            const on = Boolean(features[key]);
            return (
              <span
                key={key}
                className={`feature-pill ${on ? "feature-pill--on" : "feature-pill--off"}`}
                title={`${label}: ${on ? "enabled" : "disabled"}`}
              >
                <span className="feature-pill__dot" />
                {label}
              </span>
            );
          })}
          <span
            className="feature-pill feature-pill--info"
            title={`Vision provider: ${features.vision_provider}`}
          >
            Vision: {features.vision_provider}
          </span>
          {features.auth_enabled && features.insecure_dev_mode && (
            <span
              className="feature-pill feature-pill--warn"
              title="insecure_dev_mode: unauthenticated requests are allowed when no bearer token is supplied"
            >
              Dev mode
            </span>
          )}
        </>
      )}
      <button
        type="button"
        className="feature-flags-bar__refresh"
        onClick={refresh}
        aria-label="Refresh backend status"
        title="Refresh backend status"
      >
        ↻
      </button>
    </div>
  );
}
