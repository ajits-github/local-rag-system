"""`POST /ingest` must not block the ASGI event loop for the full ingestion duration.

Before the fix, `ingest`/`_ingest_upload_atomically` were `async def` and
called the fully synchronous `IngestionPipeline.ingest_file(...)` (disk I/O,
chunking, embedding-model inference, DB writes) directly, with no
threadpool offload -- unlike every other router, which either is a plain
`def` (auto-threadpooled by Starlette) or explicitly wraps a sync call in
`run_in_threadpool` (see `agent_stream.py`). This genuinely blocked the
single ASGI event loop for the whole ingestion duration: `/query`,
`/health`, and any open SSE stream would stall while an upload was
processing.

Uses a real `httpx.AsyncClient` bound to the app via `ASGITransport` (not
`TestClient`, which serializes requests through a blocking portal) so two
requests can genuinely run concurrently on the event loop, proving
`/health` returns well before a slow, concurrently-running `/ingest`
request completes.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import pytest

from rag.api.deps import get_config, get_ingestion_pipeline, get_llm, get_vectorstore
from rag.api.main import app
from rag.config import load_config


class _SlowFakePipeline:
    """IngestionPipeline double whose `ingest_file` blocks the calling thread briefly.

    `time.sleep` (not `asyncio.sleep`) deliberately: this simulates the real
    `IngestionPipeline.ingest_file`'s fully synchronous, CPU/IO-bound work.
    Blocking the *calling thread* is fine (and expected) as long as that
    thread isn't the event loop's own thread -- which is exactly what this
    test proves.
    """

    def __init__(self, delay_seconds: float) -> None:
        self._delay_seconds = delay_seconds
        self.calls: list[dict[str, Any]] = []

    def ingest_file(self, path, dataset_id, caller=None, source_override=None) -> dict[str, Any]:
        self.calls.append({"path": path, "dataset_id": dataset_id})
        time.sleep(self._delay_seconds)
        return {"document_id": "doc-1", "chunks_written": 1, "changed": True}


class _FakeVectorStore:
    """Minimal VectorStore double that only answers health checks, instantly."""

    def health_check(self) -> bool:
        """Report healthy, always, with no real I/O."""
        return True


class _FakeLLM:
    """Minimal LLM double that only answers health checks, instantly."""

    def health_check(self) -> bool:
        """Report healthy, always, with no real I/O."""
        return True


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    """Redirect `UPLOAD_DIR` to an isolated `tmp_path`, matching the established test pattern."""
    monkeypatch.setattr("rag.api.routers.ingest.UPLOAD_DIR", tmp_path)
    return tmp_path


@pytest.mark.asyncio
async def test_slow_ingest_does_not_block_a_concurrent_health_request(upload_dir):
    """A concurrently-running slow /ingest never delays /health's response.

    Fires both requests at once via `asyncio.gather`; if `/ingest` still
    blocked the event loop, `/health` could not complete until `/ingest`
    finished, and its measured latency would be close to `delay_seconds`
    rather than near-instant.
    """
    delay_seconds = 1.0
    pipeline = _SlowFakePipeline(delay_seconds)
    config = load_config()
    app.dependency_overrides[get_config] = lambda: config
    app.dependency_overrides[get_ingestion_pipeline] = lambda: pipeline
    app.dependency_overrides[get_vectorstore] = lambda: _FakeVectorStore()
    app.dependency_overrides[get_llm] = lambda: _FakeLLM()
    # Captured *before* `asyncio.gather` starts either coroutine, so it's a fixed
    # reference point for "wall clock time since the two requests were issued" --
    # not time measured from inside a coroutine that might itself be starved of
    # a chance to run until the blocking call releases the loop (which would
    # misleadingly read as "fast" regardless of whether the fix is in place).
    t_start = time.monotonic()
    health_done_at: list[float] = []
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

            async def _do_ingest() -> httpx.Response:
                return await client.post(
                    "/ingest",
                    data={"dataset_id": "test-dataset"},
                    files={"files": ("a.md", b"# hello", "text/markdown")},
                )

            async def _do_health_after_a_beat() -> httpx.Response:
                await asyncio.sleep(0.1)  # let the ingest request actually start first
                response = await client.get("/health")
                health_done_at.append(time.monotonic() - t_start)
                return response

            ingest_response, health_response = await asyncio.gather(
                _do_ingest(), _do_health_after_a_beat()
            )
    finally:
        app.dependency_overrides.pop(get_config, None)
        app.dependency_overrides.pop(get_ingestion_pipeline, None)
        app.dependency_overrides.pop(get_vectorstore, None)
        app.dependency_overrides.pop(get_llm, None)

    assert ingest_response.status_code == 200
    assert pipeline.calls == [{"path": pipeline.calls[0]["path"], "dataset_id": "test-dataset"}]
    assert health_response.status_code == 200
    # /health returned well before the 1s ingest finished -- it was never
    # stuck behind it on the event loop.
    assert health_done_at[0] < delay_seconds / 2
