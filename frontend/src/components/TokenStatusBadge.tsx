import { useEffect, useRef, useState } from "react";
import { useTokenStatus, type TokenStatusState } from "../hooks/useTokenStatus";

const STATE_TEXT: Record<TokenStatusState, (countdown: string | null) => string> = {
  none: () => "",
  invalid: () => "Invalid token",
  no_exp: () => "Authenticated",
  healthy: (countdown) => `Authenticated · ${countdown}`,
  expiring_soon: (countdown) => `Authenticated · ${countdown}`,
  critical: (countdown) => `Token expiring · ${countdown}`,
  expired: () => "Token expired",
};

// Decorative only (aria-hidden); state is always conveyed by the text
// alone too, per the "don't rely on color alone" requirement.
const STATE_ICON: Record<TokenStatusState, string> = {
  none: "",
  invalid: "⚠",
  no_exp: "●",
  healthy: "●",
  expiring_soon: "◐",
  critical: "⚠",
  expired: "✕",
};

// Announced once per state transition, not every countdown tick -- a
// live region that updated every second would spam assistive tech.
const ANNOUNCE_TEXT: Partial<Record<TokenStatusState, string>> = {
  expiring_soon: "Developer session token is expiring soon.",
  critical: "Developer session token is about to expire.",
  expired: "Developer session token has expired.",
  invalid: "Developer session token is not a valid JWT.",
};

/**
 * Header countdown/status indicator for the developer bearer token (see
 * `21.0-UI-improvement`'s "JWT expiration UX" requirement). Decoding is
 * display-only -- see utils/jwt.ts -- and never implies the token is
 * actually valid server-side; backend verification remains authoritative.
 */
export function TokenStatusBadge({
  token,
  onReplaceToken,
}: {
  token: string;
  onReplaceToken: () => void;
}) {
  const status = useTokenStatus(token);
  const [announcement, setAnnouncement] = useState("");
  const lastAnnouncedState = useRef<TokenStatusState | null>(null);

  useEffect(() => {
    if (status.state === lastAnnouncedState.current) return;
    lastAnnouncedState.current = status.state;
    const text = ANNOUNCE_TEXT[status.state];
    if (text) setAnnouncement(text);
  }, [status.state]);

  if (!status.hasToken) return null;

  const text = STATE_TEXT[status.state](status.countdownLabel);
  const needsReplace = status.state === "expired" || status.state === "invalid";

  return (
    <div className={`token-status token-status--${status.state}`}>
      <span
        className="token-status__pill"
        title={status.expiresAtLabel ? `Token expires ${status.expiresAtLabel}` : undefined}
      >
        <span aria-hidden="true" className="token-status__icon">
          {STATE_ICON[status.state]}
        </span>
        {text}
      </span>
      {needsReplace && (
        <button type="button" className="token-status__replace" onClick={onReplaceToken}>
          Replace token
        </button>
      )}
      <span className="sr-only" role="status" aria-live="polite">
        {announcement}
      </span>
    </div>
  );
}
