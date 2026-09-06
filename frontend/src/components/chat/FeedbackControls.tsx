import { useState } from "react";
import { postFeedback, type FeedbackRequestBody } from "../../api/feedback";
import { RagApiError } from "../../api/client";
import { NEGATIVE_FEEDBACK_REASONS, POSITIVE_FEEDBACK_REASONS, type DevIdentity, type FeedbackRating } from "../../api/types";
import { useChat } from "../../state/chatContext";
import type { ChatMessage, FeedbackState } from "../../state/types";

/** Soft client-side guard only; the backend (FeedbackConfig.max_comment_length) enforces the real bound. */
const MAX_COMMENT_LENGTH = 1000;

function buildBody(
  message: ChatMessage,
  rating: FeedbackRating,
  reason: string,
  comment: string,
  identity: DevIdentity
): FeedbackRequestBody {
  const body: FeedbackRequestBody = {
    request_id: message.debug?.requestId ?? "",
    rating,
    reason: reason || undefined,
    comment: comment.trim() || undefined,
    query: message.query,
    answer: message.text,
    route: message.debug?.route,
    cited_source_ids: message.sources.map((s) => s.chunk_id),
    tool_calls: message.debug?.toolCalls,
  };
  // Mirrors buildQueryRequestBody/buildAgentQueryRequestBody: only sent when
  // no bearer token is present (the backend ignores it otherwise). Without
  // this, a tenant-scoped dev-identity query's feedback would be stored
  // under the no-tenant sentinel instead of that tenant.
  if (!identity.bearerToken && identity.tenantId) {
    body.tenant_id = identity.tenantId;
  }
  return body;
}

/**
 * Thumbs up/down + optional reason/comment for one completed assistant answer.
 *
 * Clicking the already-selected, already-submitted rating again is a no-op
 * (not an "explicit change"); clicking the other rating, or submitting the
 * details form, replaces the earlier submission via POST /feedback's own
 * update semantics (same request_id -> same row).
 */
export function FeedbackControls({ message }: { message: ChatMessage }) {
  const { state, dispatch } = useChat();
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [reason, setReason] = useState("");
  const [comment, setComment] = useState("");

  const requestId = message.debug?.requestId;
  const feedback: FeedbackState = message.feedback ?? { status: "idle" };

  if (!requestId) return null;

  const setFeedback = (next: FeedbackState) => dispatch({ type: "SET_FEEDBACK", id: message.id, feedback: next });

  async function submit(rating: FeedbackRating, withDetails: boolean) {
    if (feedback.status === "submitted" && feedback.rating === rating && !withDetails) {
      return;
    }
    setFeedback({ status: "submitting", rating });
    const body = buildBody(
      message,
      rating,
      withDetails ? reason : feedback.rating === rating ? (feedback.reason ?? "") : "",
      withDetails ? comment : feedback.rating === rating ? (feedback.comment ?? "") : "",
      state.devIdentity
    );
    try {
      const ack = await postFeedback(body, state.devIdentity);
      setFeedback({
        status: "submitted",
        rating,
        feedbackId: ack.feedback_id,
        reason: body.reason,
        comment: body.comment,
      });
      if (withDetails) setDetailsOpen(false);
    } catch (err) {
      const errorMessage = err instanceof RagApiError ? err.message : "Could not submit feedback.";
      setFeedback({ status: "error", rating, errorMessage });
    }
  }

  const reasonOptions = feedback.rating === "positive" ? POSITIVE_FEEDBACK_REASONS : NEGATIVE_FEEDBACK_REASONS;
  const isSubmitting = feedback.status === "submitting";
  const hasRating = feedback.status === "submitted" || feedback.status === "error";

  return (
    <div className="feedback-controls">
      <div className="feedback-controls__buttons" role="group" aria-label="Rate this answer">
        <button
          type="button"
          className="feedback-button"
          aria-pressed={feedback.rating === "positive" && feedback.status === "submitted"}
          aria-label="Helpful answer"
          disabled={isSubmitting}
          onClick={() => void submit("positive", false)}
        >
          👍
        </button>
        <button
          type="button"
          className="feedback-button"
          aria-pressed={feedback.rating === "negative" && feedback.status === "submitted"}
          aria-label="Not helpful answer"
          disabled={isSubmitting}
          onClick={() => void submit("negative", false)}
        >
          👎
        </button>
        {hasRating && !detailsOpen && (
          <button type="button" className="feedback-controls__details-toggle" onClick={() => setDetailsOpen(true)}>
            Add details
          </button>
        )}
      </div>

      {detailsOpen && feedback.rating && (
        <form
          className="feedback-controls__details"
          onSubmit={(e) => {
            e.preventDefault();
            void submit(feedback.rating as FeedbackRating, true);
          }}
        >
          <label>
            Reason (optional)
            <select value={reason} onChange={(e) => setReason(e.target.value)}>
              <option value="">Select a reason</option>
              {reasonOptions.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Comment (optional)
            <textarea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              maxLength={MAX_COMMENT_LENGTH}
              rows={2}
            />
          </label>
          <div className="feedback-controls__details-actions">
            <button type="submit" disabled={isSubmitting}>
              Submit
            </button>
            <button type="button" onClick={() => setDetailsOpen(false)}>
              Cancel
            </button>
          </div>
        </form>
      )}

      <p className="feedback-controls__status" role="status" aria-live="polite">
        {feedback.status === "submitted" && "Thanks for your feedback."}
        {feedback.status === "error" && feedback.rating && (
          <>
            {feedback.errorMessage ?? "Could not submit feedback."}{" "}
            <button type="button" onClick={() => void submit(feedback.rating as FeedbackRating, detailsOpen)}>
              Retry
            </button>
          </>
        )}
      </p>
    </div>
  );
}
