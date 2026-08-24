import { expect, test } from "@playwright/test";

/**
 * A small set of end-to-end flows, run against the real Vite dev server
 * with the backend mocked at the network layer (page.route intercepts
 * before any request leaves the browser, so no live rag-api/Postgres/
 * Ollama is required). Keep this file to the handful of flows that only a
 * real browser can verify (routing, streaming reads, focus/scroll); every
 * other behavior is covered by the Vitest integration tests instead.
 */

test.describe("critical chat flows", () => {
  test("classic RAG: ask a question, see the answer and expand its sources", async ({ page }) => {
    await page.route("**/query", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          answer: "Passwords must be at least 12 characters.",
          sources: [
            {
              chunk_id: "c1",
              document_id: "d1",
              source: "knowledge_base/security/password-policy.md",
              category: "security",
              score: 0.93,
              content_type: "prose",
              section_path: "Requirements",
              page: null,
              attachment_name: null,
              source_anchor: null,
              vision_generated: false,
            },
          ],
          retrieval_ms: 10,
          generation_ms: 200,
          total_ms: 210,
        }),
      });
    });

    await page.goto("/");
    await page.getByLabel("Message").fill("What is the password policy?");
    await page.getByRole("button", { name: "Send" }).click();

    await expect(page.getByText("Passwords must be at least 12 characters.")).toBeVisible();

    await page.getByRole("button", { name: /Sources \(1\)/ }).click();
    await expect(page.getByText("knowledge_base/security/password-policy.md")).toBeVisible();
  });

  test("agentic RAG: switch mode, watch live activity, then see the final answer", async ({ page }) => {
    await page.route("**/agent/query/stream", async (route) => {
      const frames = [
        `event: route_selected\ndata: ${JSON.stringify({ event_type: "route_selected", route: "agent" })}\n\n`,
        `event: tool_started\ndata: ${JSON.stringify({
          event_type: "tool_started",
          tool_name: "search_knowledge_base",
          step: 1,
        })}\n\n`,
        `event: tool_completed\ndata: ${JSON.stringify({
          event_type: "tool_completed",
          tool_name: "search_knowledge_base",
          retrieved_chunk_count: 3,
          step: 1,
        })}\n\n`,
        `event: completed\ndata: ${JSON.stringify({
          answer: "The agent combined two documents to answer this.",
          sources: [],
          route: "agent",
          termination_reason: "synthesized",
          steps: 2,
          tool_calls: ["search_knowledge_base"],
          retrieval_ms: 30,
          generation_ms: 600,
          total_ms: 630,
        })}\n\n`,
      ].join("");

      await route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        body: frames,
      });
    });

    await page.goto("/");
    await page.getByRole("radio", { name: "Agentic RAG" }).click();
    await page.getByLabel("Message").fill("How do ingestion and retrieval fit together?");
    await page.getByRole("button", { name: "Send" }).click();

    await expect(page.getByText(/search_knowledge_base/).first()).toBeVisible();
    await expect(page.getByText("The agent combined two documents to answer this.")).toBeVisible();
  });

  test("shows a clear error, not a fake answer, when the backend is unreachable", async ({ page }) => {
    await page.route("**/query", async (route) => {
      await route.abort("connectionrefused");
    });

    await page.goto("/");
    await page.getByLabel("Message").fill("Anything");
    await page.getByRole("button", { name: "Send" }).click();

    await expect(page.getByRole("alert")).toContainText("Could not reach the backend");
  });
});
