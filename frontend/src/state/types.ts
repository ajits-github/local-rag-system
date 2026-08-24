import type { AgentEvent, AgentQueryResponse, QueryResponse, RagMode, SourceItem, TerminationReason } from "../api/types";
import type { RagApiErrorKind } from "../api/client";

export interface DebugInfo {
  route?: "classic_rag" | "agent";
  steps?: number;
  toolCalls?: string[];
  terminationReason?: TerminationReason | null;
  retrievalMs?: number;
  generationMs?: number;
  totalMs?: number;
}

export type MessageStatus = "pending" | "streaming" | "done" | "error";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  mode: RagMode;
  status: MessageStatus;
  text: string;
  sources: SourceItem[];
  agentEvents: AgentEvent[];
  debug?: DebugInfo;
  errorKind?: RagApiErrorKind;
  errorMessage?: string;
  insufficientEvidence?: boolean;
  streamFellBack?: boolean;
  createdAt: number;
}

export function debugFromQueryResponse(response: QueryResponse): DebugInfo {
  return {
    route: "classic_rag",
    retrievalMs: response.retrieval_ms,
    generationMs: response.generation_ms,
    totalMs: response.total_ms,
  };
}

export function debugFromAgentResponse(response: AgentQueryResponse): DebugInfo {
  return {
    route: response.route,
    steps: response.steps,
    toolCalls: response.tool_calls,
    terminationReason: response.termination_reason,
    retrievalMs: response.retrieval_ms,
    generationMs: response.generation_ms,
    totalMs: response.total_ms,
  };
}
