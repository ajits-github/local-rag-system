import { useCallback, useEffect, useRef, useState } from "react";
import { fetchRootInfo } from "../api/rootInfo";
import type { FeatureFlags } from "../api/types";

interface BackendFeaturesState {
  features: FeatureFlags | null;
  loading: boolean;
  error: string | null;
}

// Short backoff for the automatic mount-time check only: a container
// stack brought up together (e.g. `docker compose up -d`, or the
// nginx/rag-api recreate that follows an env-var/image change) can take
// a few seconds after this page's own load before rag-api is accepting
// connections. A single unretried fetch can lose that race and land on
// "Backend status unavailable" even though the backend comes up
// moments later and every subsequent query succeeds -- there was
// previously no automatic recovery from that, only the manual refresh
// button. A manually-triggered refresh() stays single-shot: a user
// clicking the button already knows to click again if it fails.
const MOUNT_RETRY_DELAYS_MS = [500, 1500, 3000];

export function useBackendFeatures() {
  const [state, setState] = useState<BackendFeaturesState>({
    features: null,
    loading: true,
    error: null,
  });
  const abortRef = useRef<AbortController | null>(null);

  const refresh = useCallback(() => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setState((s) => ({ ...s, loading: true, error: null }));
    fetchRootInfo(controller.signal)
      .then((info) => setState({ features: info.features, loading: false, error: null }))
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setState({
          features: null,
          loading: false,
          error: err instanceof Error ? err.message : "Failed to load backend status.",
        });
      });
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    abortRef.current = controller;
    let cancelled = false;

    async function loadWithRetry() {
      for (let attempt = 0; ; attempt++) {
        setState((s) => ({ ...s, loading: true, error: null }));
        try {
          const info = await fetchRootInfo(controller.signal);
          if (!cancelled) setState({ features: info.features, loading: false, error: null });
          return;
        } catch (err) {
          if (cancelled || controller.signal.aborted) return;
          if (attempt >= MOUNT_RETRY_DELAYS_MS.length) {
            setState({
              features: null,
              loading: false,
              error: err instanceof Error ? err.message : "Failed to load backend status.",
            });
            return;
          }
          await new Promise((resolve) => setTimeout(resolve, MOUNT_RETRY_DELAYS_MS[attempt]));
        }
      }
    }

    void loadWithRetry();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, []);

  return { ...state, refresh };
}
