/**
 * Zod schemas mirroring the backend's Pydantic response models
 * (src/rag/api/routers/query.py, agent_query.py, agent_stream.py,
 * src/rag/agent/events.py). Parsing every response through these schemas
 * is what lets the UI treat "malformed server response" as a distinct,
 * handleable error state instead of crashing on an unexpected shape.
 */
import { z } from "zod";

export const SourceItemSchema = z.object({
  chunk_id: z.string(),
  document_id: z.string(),
  source: z.string(),
  category: z.string().nullable().optional(),
  score: z.number(),
  content_type: z.string().nullable().optional(),
  section_path: z.string().nullable().optional(),
  page: z.number().nullable().optional(),
  attachment_name: z.string().nullable().optional(),
  source_anchor: z.string().nullable().optional(),
  vision_generated: z.boolean().default(false),
});
export type SourceItem = z.infer<typeof SourceItemSchema>;

export const QueryResponseSchema = z.object({
  answer: z.string(),
  sources: z.array(SourceItemSchema),
  retrieval_ms: z.number(),
  generation_ms: z.number(),
  total_ms: z.number(),
});
export type QueryResponse = z.infer<typeof QueryResponseSchema>;

export const TerminationReasonSchema = z.enum([
  "synthesized",
  "max_steps",
  "max_retrieval_attempts",
  "max_tool_calls",
  "insufficient_evidence",
]);
export type TerminationReason = z.infer<typeof TerminationReasonSchema>;

export const AgentQueryResponseSchema = z.object({
  answer: z.string(),
  sources: z.array(SourceItemSchema),
  route: z.enum(["classic_rag", "agent"]),
  termination_reason: TerminationReasonSchema.nullable(),
  steps: z.number(),
  tool_calls: z.array(z.string()),
  retrieval_ms: z.number(),
  generation_ms: z.number(),
  total_ms: z.number(),
});
export type AgentQueryResponse = z.infer<typeof AgentQueryResponseSchema>;

export const AgentEventTypeSchema = z.enum([
  "query_received",
  "route_selected",
  "decomposition_started",
  "decomposition_completed",
  "tool_selected",
  "tool_started",
  "tool_completed",
  "evidence_evaluated",
  "retry_started",
  "synthesis_started",
  "completed",
  "terminated",
]);
export type AgentEventType = z.infer<typeof AgentEventTypeSchema>;

export const AgentEventSchema = z.object({
  event_type: AgentEventTypeSchema,
  step: z.number().nullable().optional(),
  tool_name: z.string().nullable().optional(),
  elapsed_ms: z.number().nullable().optional(),
  retrieved_chunk_count: z.number().nullable().optional(),
  evidence_sufficient: z.boolean().nullable().optional(),
  termination_reason: z.string().nullable().optional(),
  route: z.string().nullable().optional(),
});
export type AgentEvent = z.infer<typeof AgentEventSchema>;

export const FeatureFlagsSchema = z.object({
  auth_enabled: z.boolean(),
  insecure_dev_mode: z.boolean(),
  authorization_enabled: z.boolean(),
  field_redaction_enabled: z.boolean(),
  rate_limit_enabled: z.boolean(),
  agent_enabled: z.boolean(),
  vision_provider: z.string(),
  tracing_enabled: z.boolean(),
});
export type FeatureFlags = z.infer<typeof FeatureFlagsSchema>;

export const RootInfoSchema = z.object({
  service: z.string(),
  status: z.string(),
  docs: z.string(),
  health: z.string(),
  metrics: z.string().nullable(),
  features: FeatureFlagsSchema,
});
export type RootInfo = z.infer<typeof RootInfoSchema>;

export type RagMode = "classic" | "agent";

/** Local-development-only identity a caller may assert (see DevIdentityPanel). */
export interface DevIdentity {
  bearerToken: string;
  tenantId: string;
  roles: string;
  asOf: string;
  requireTrustLevel: string;
}
