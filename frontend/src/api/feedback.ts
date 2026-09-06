import { postJson, RagApiError } from "./client";
import { FeedbackAckSchema, type FeedbackAck, type FeedbackRating, type DevIdentity } from "./types";

export interface FeedbackRequestBody {
  request_id: string;
  rating: FeedbackRating;
  reason?: string;
  comment?: string;
  query?: string;
  answer?: string;
  route?: "classic_rag" | "agent";
  dataset_id?: string;
  cited_source_ids?: string[];
  tool_calls?: string[];
  tenant_id?: string;
}

/** POST /feedback. Reuses postJson's error classification (client.ts) unchanged. */
export async function postFeedback(body: FeedbackRequestBody, identity?: DevIdentity): Promise<FeedbackAck> {
  const response = await postJson("/feedback", body, { identity });
  const json = await response.json();
  const parsed = FeedbackAckSchema.safeParse(json);
  if (!parsed.success) {
    throw new RagApiError("malformed_response", "The server returned an unexpected feedback response shape.");
  }
  return parsed.data;
}
