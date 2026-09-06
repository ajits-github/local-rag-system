import { useCallback, useRef, useState } from "react";
import { fetchRuntimeInfo } from "../api/runtimeInfo";
import type { RuntimeInfo } from "../api/types";

interface RuntimeInfoState {
  info: RuntimeInfo | null;
  loading: boolean;
  error: string | null;
  loaded: boolean;
}

/**
 * Fetches GET /info on demand, not on mount: unlike useBackendFeatures
 * (which must always be visible for security awareness), runtime
 * configuration is opt-in developer/debug detail shown inside a
 * collapsed-by-default panel, so there is no reason to spend a request
 * on it before a caller actually expands that panel.
 */
export function useRuntimeInfo() {
  const [state, setState] = useState<RuntimeInfoState>({
    info: null,
    loading: false,
    error: null,
    loaded: false,
  });
  const abortRef = useRef<AbortController | null>(null);

  const load = useCallback(() => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setState((s) => ({ ...s, loading: true, error: null }));
    fetchRuntimeInfo(controller.signal)
      .then((info) => setState({ info, loading: false, error: null, loaded: true }))
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setState({
          info: null,
          loading: false,
          error: err instanceof Error ? err.message : "Failed to load runtime configuration.",
          loaded: true,
        });
      });
  }, []);

  return { ...state, load };
}
