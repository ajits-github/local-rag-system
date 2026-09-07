import { useCallback, useEffect, useRef, useState } from "react";
import { fetchRuntimeInfo } from "../api/runtimeInfo";
import type { RuntimeInfo } from "../api/types";

interface RuntimeInfoState {
  info: RuntimeInfo | null;
  loading: boolean;
  error: string | null;
  loaded: boolean;
}

/**
 * Fetches GET /info once on mount, then again only on an explicit
 * `load()` call (the panel's "Refresh" button).
 *
 * Loaded eagerly, unlike a purely on-demand fetch, because the compact
 * runtime badge (RuntimeInfoPanel's collapsed-state summary, e.g. "qwen2.5:3b
 * · Hybrid · RRF · MCP") needs this data to render at all -- the same
 * "always know what the connected backend is actually running" rationale
 * useBackendFeatures already applies to GET /'s FeatureFlags. The detailed
 * field-by-field breakdown stays behind the panel's own collapsible toggle.
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

  useEffect(() => {
    load();
    return () => abortRef.current?.abort();
  }, [load]);

  return { ...state, load };
}
