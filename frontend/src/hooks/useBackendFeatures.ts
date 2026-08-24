import { useCallback, useEffect, useRef, useState } from "react";
import { fetchRootInfo } from "../api/rootInfo";
import type { FeatureFlags } from "../api/types";

interface BackendFeaturesState {
  features: FeatureFlags | null;
  loading: boolean;
  error: string | null;
}

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
    refresh();
    return () => abortRef.current?.abort();
  }, [refresh]);

  return { ...state, refresh };
}
