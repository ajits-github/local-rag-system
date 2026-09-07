import type {
  AgentEvent,
  AgentQueryResponse,
  FeedbackRating,
  QueryResponse,
  RagMode,
  SourceItem,
  TerminationReason,
  ToolCallDetail,
} from "../api/types";
import type { RagApiErrorKind } from "../api/client";

export interface DebugInfo {
  route?: "classic_rag" | "agent";
  steps?: number;
  toolCalls?: string[];
  toolCallDetails?: ToolCallDetail[];
  terminationReason?: TerminationReason | null;
  retrievalMs?: number;
  generationMs?: number;
  totalMs?: number;
  /** Correlation id from the response (see QueryResponse.request_id); the answer/run a feedback submission ties to. */
  requestId?: string;
}

export type MessageStatus = "pending" | "streaming" | "done" | "error";

/** Local state for one message's feedback controls; mirrors POST /feedback's own rating/reason/comment shape. */
export interface FeedbackState {
  status: "idle" | "submitting" | "submitted" | "error";
  rating?: FeedbackRating;
  reason?: string;
  comment?: string;
  feedbackId?: string;
  errorMessage?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  mode: RagMode;
  status: MessageStatus;
  text: string;
  /** The original question text, set on the assistant message once it completes (feedback needs both sides). */
  query?: string;
  sources: SourceItem[];
  agentEvents: AgentEvent[];
  debug?: DebugInfo;
  feedback?: FeedbackState;
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
    requestId: response.request_id ?? undefined,
  };
}

export function debugFromAgentResponse(response: AgentQueryResponse): DebugInfo {
  return {
    route: response.route,
    steps: response.steps,
    toolCalls: response.tool_calls,
    toolCallDetails: response.tool_call_details,
    terminationReason: response.termination_reason,
    retrievalMs: response.retrieval_ms,
    generationMs: response.generation_ms,
    totalMs: response.total_ms,
    requestId: response.request_id ?? undefined,
  };
}
