import { useEffect } from "react";
import userEvent from "@testing-library/user-event";
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ChatProvider, useChat } from "../../src/state/chatContext";
import { FeedbackControls } from "../../src/components/chat/FeedbackControls";
import type { SourceItem } from "../../src/api/types";
import type { DebugInfo } from "../../src/state/types";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

/** Seeds one completed assistant message into ChatProvider's real reducer, then renders FeedbackControls wired to it. */
function Harness({ debug, sources = [] }: { debug: DebugInfo; sources?: SourceItem[] }) {
  const { state, dispatch } = useChat();
  useEffect(() => {
    dispatch({ type: "ADD_ASSISTANT_PLACEHOLDER", id: "a1", mode: "classic" });
    dispatch({
      type: "COMPLETE_ASSISTANT_ANSWER",
      id: "a1",
      query: "what is the retention policy?",
      text: "The retention policy is 90 days.",
      sources,
      debug,
      insufficientEvidence: false,
    });
    // Run once on mount only; this harness seeds a fixed message, it never needs to reseed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const message = state.messages[0];
  if (!message) return null;
  return <FeedbackControls message={message} />;
}

function renderFeedback(debug: DebugInfo = { route: "classic_rag", requestId: "req-1" }, sources: SourceItem[] = []) {
  return render(
    <ChatProvider>
      <Harness debug={debug} sources={sources} />
    </ChatProvider>
  );
}

describe("FeedbackControls", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    window.sessionStorage.clear();
  });

  it("renders accessible thumbs-up/thumbs-down controls", async () => {
    renderFeedback();
    expect(await screen.findByRole("button", { name: "Helpful answer" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Not helpful answer" })).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "Rate this answer" })).toBeInTheDocument();
  });

  it("does not render when the message has no request_id (nothing to correlate feedback to)", () => {
    renderFeedback({ route: "classic_rag" });
    expect(screen.queryByRole("button", { name: "Helpful answer" })).not.toBeInTheDocument();
  });

  it("submits a positive rating with the expected payload", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { feedback_id: "fb-1", status: "created" }));
    vi.stubGlobal("fetch", fetchMock);
    const sources: SourceItem[] = [
      { chunk_id: "doc_0", document_id: "doc", source: "a.md", score: 0.9, vision_generated: false, origin: "retrieved" },
    ];
    renderFeedback({ route: "classic_rag", requestId: "req-1", toolCalls: ["search_knowledge_base"] }, sources);

    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Helpful answer" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    expect(body).toMatchObject({
      request_id: "req-1",
      rating: "positive",
      query: "what is the retention policy?",
      answer: "The retention policy is 90 days.",
      route: "classic_rag",
      cited_source_ids: ["doc_0"],
      tool_calls: ["search_knowledge_base"],
    });
    expect(await screen.findByText("Thanks for your feedback.")).toBeInTheDocument();
  });

  it("does not resubmit when the same already-submitted rating is clicked again", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { feedback_id: "fb-1", status: "created" }));
    vi.stubGlobal("fetch", fetchMock);
    renderFeedback();
    const user = userEvent.setup();

    const thumbsUp = await screen.findByRole("button", { name: "Helpful answer" });
    await user.click(thumbsUp);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    await user.click(thumbsUp);
    // Give any accidental async re-submission a chance to fire before asserting it didn't.
    await new Promise((r) => setTimeout(r, 10));
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("replaces an earlier rating when the user explicitly changes it", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(200, { feedback_id: "fb-1", status: "created" }))
      .mockResolvedValueOnce(jsonResponse(200, { feedback_id: "fb-1", status: "updated" }));
    vi.stubGlobal("fetch", fetchMock);
    renderFeedback();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "Helpful answer" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    await user.click(screen.getByRole("button", { name: "Not helpful answer" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));

    const secondBody = JSON.parse((fetchMock.mock.calls[1][1] as RequestInit).body as string);
    expect(secondBody.rating).toBe("negative");
  });

  it("submits an optional reason and comment as a follow-up update", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(200, { feedback_id: "fb-1", status: "created" }))
      .mockResolvedValueOnce(jsonResponse(200, { feedback_id: "fb-1", status: "updated" }));
    vi.stubGlobal("fetch", fetchMock);
    renderFeedback();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "Not helpful answer" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    await user.click(screen.getByRole("button", { name: "Add details" }));
    await user.selectOptions(screen.getByLabelText("Reason (optional)"), "wrong_or_missing_citation");
    await user.type(screen.getByLabelText("Comment (optional)"), "cited the wrong document");
    await user.click(screen.getByRole("button", { name: "Submit" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    const secondBody = JSON.parse((fetchMock.mock.calls[1][1] as RequestInit).body as string);
    expect(secondBody).toMatchObject({
      rating: "negative",
      reason: "wrong_or_missing_citation",
      comment: "cited the wrong document",
    });
  });

  it("includes tenant_id from the dev identity when no bearer token is present", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { feedback_id: "fb-1", status: "created" }));
    vi.stubGlobal("fetch", fetchMock);

    function TenantHarness({ debug }: { debug: DebugInfo }) {
      const { state, dispatch } = useChat();
      useEffect(() => {
        dispatch({
          type: "SET_DEV_IDENTITY",
          identity: {
            bearerToken: "",
            tenantId: "tenant_alpha",
            roles: "",
            asOf: "",
            requireTrustLevel: "",
            datasetId: "",
          },
        });
        dispatch({ type: "ADD_ASSISTANT_PLACEHOLDER", id: "a1", mode: "classic" });
        dispatch({
          type: "COMPLETE_ASSISTANT_ANSWER",
          id: "a1",
          query: "what is the retention policy?",
          text: "The retention policy is 90 days.",
          sources: [],
          debug,
          insufficientEvidence: false,
        });
        // eslint-disable-next-line react-hooks/exhaustive-deps
      }, []);
      const message = state.messages[0];
      if (!message) return null;
      return <FeedbackControls message={message} />;
    }

    render(
      <ChatProvider>
        <TenantHarness debug={{ route: "classic_rag", requestId: "req-1" }} />
      </ChatProvider>
    );

    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Helpful answer" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    expect(body.tenant_id).toBe("tenant_alpha");
  });

  it("omits tenant_id when a bearer token is present (the backend ignores it either way)", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { feedback_id: "fb-1", status: "created" }));
    vi.stubGlobal("fetch", fetchMock);

    function TenantHarness({ debug }: { debug: DebugInfo }) {
      const { state, dispatch } = useChat();
      useEffect(() => {
        dispatch({
          type: "SET_DEV_IDENTITY",
          identity: {
            bearerToken: "some-jwt",
            tenantId: "tenant_alpha",
            roles: "",
            asOf: "",
            requireTrustLevel: "",
            datasetId: "",
          },
        });
        dispatch({ type: "ADD_ASSISTANT_PLACEHOLDER", id: "a1", mode: "classic" });
        dispatch({
          type: "COMPLETE_ASSISTANT_ANSWER",
          id: "a1",
          query: "what is the retention policy?",
          text: "The retention policy is 90 days.",
          sources: [],
          debug,
          insufficientEvidence: false,
        });
        // eslint-disable-next-line react-hooks/exhaustive-deps
      }, []);
      const message = state.messages[0];
      if (!message) return null;
      return <FeedbackControls message={message} />;
    }

    render(
      <ChatProvider>
        <TenantHarness debug={{ route: "classic_rag", requestId: "req-1" }} />
      </ChatProvider>
    );

    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Helpful answer" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    expect(body.tenant_id).toBeUndefined();
  });

  it("shows a retry affordance when submission fails", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));
    vi.stubGlobal("fetch", fetchMock);
    renderFeedback();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "Helpful answer" }));

    expect(await screen.findByRole("button", { name: "Retry" })).toBeInTheDocument();
  });
});
