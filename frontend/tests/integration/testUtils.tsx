import { render } from "@testing-library/react";
import { vi } from "vitest";
import { ChatProvider } from "../../src/state/chatContext";
import { ChatWindow } from "../../src/components/chat/ChatWindow";
import type { FeatureFlags, RuntimeInfo } from "../../src/api/types";

export function renderChatWindow() {
  return render(
    <ChatProvider>
      <ChatWindow />
    </ChatProvider>
  );
}

export function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/** A valid GET / response; ChatWindow's FeatureFlagsBar fetches this on mount. */
export function rootInfoResponse(featureOverrides: Partial<FeatureFlags> = {}): Response {
  return jsonResponse(200, {
    service: "local-rag-system",
    status: "ok",
    docs: "/docs",
    health: "/health",
    metrics: "/metrics",
    features: {
      auth_enabled: false,
      insecure_dev_mode: false,
      authorization_enabled: false,
      field_redaction_enabled: false,
      rate_limit_enabled: false,
      agent_enabled: true,
      vision_provider: "none",
      tracing_enabled: false,
      ...featureOverrides,
    },
  });
}

/** A valid GET /info response; ChatWindow's RuntimeInfoPanel fetches this on mount (for its compact badge). */
export function runtimeInfoResponse(overrides: Partial<RuntimeInfo> = {}): Response {
  return jsonResponse(200, {
    generation_provider: "ollama",
    generation_model: "qwen2.5:3b",
    embedding_provider: "sentence_transformers",
    embedding_model: "sentence-transformers/all-MiniLM-L6-v2",
    vectorstore_provider: "pgvector",
    retrieval_mode: "hybrid",
    sparse_retrieval_enabled: true,
    fusion_method: "rrf",
    reranker_provider: "none",
    reranker_enabled: false,
    agent_enabled: true,
    mcp_enabled: false,
    mcp_client_enabled: false,
    vision_provider: "none",
    tracing_enabled: false,
    rate_limit_enabled: false,
    field_redaction_enabled: false,
    authorization_enabled: false,
    auth_enabled: false,
    ...overrides,
  });
}

/**
 * Build a `fetch` mock that dispatches by exact URL. Every ChatWindow
 * render fires a `GET /` (FeatureFlagsBar) and a `GET /info`
 * (RuntimeInfoPanel's compact badge) on mount, so this defaults both
 * unless a test explicitly overrides them. This keeps every other test's
 * fetch-call assertions from tripping over those two extra calls.
 */
export function routedFetchMock(routes: Record<string, () => Response | Promise<Response>>) {
  const allRoutes: Record<string, () => Response | Promise<Response>> = {
    "/": () => rootInfoResponse(),
    "/info": () => runtimeInfoResponse(),
    ...routes,
  };
  return vi.fn(async (url: string, _init?: RequestInit) => {
    const handler = allRoutes[url];
    if (!handler) {
      throw new Error(`Unmocked fetch call to ${url}`);
    }
    return handler();
  });
}

/** Build a fetch Response-like object streaming the given SSE frame strings. */
export function sseResponse(frameChunks: string[]): Response {
  const encoder = new TextEncoder();
  let i = 0;
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (i < frameChunks.length) {
        controller.enqueue(encoder.encode(frameChunks[i]));
        i += 1;
      } else {
        controller.close();
      }
    },
  });
  return { ok: true, status: 200, body } as unknown as Response;
}

export function sseFrame(eventType: string, data: unknown): string {
  return `event: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`;
}
