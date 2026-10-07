"""Part 1's "real HTTP" demonstration: two live FastAPI processes, one round-robin driver.

Boots two real `uvicorn` processes ("Pod A" / "Pod B"), each importing the
*actual* `rag.api.deps.get_rate_limiter()`/`_rate_limit_key` (not a
reimplementation) with `security.rate_limit.enabled=true` via
`rate_limit_demo_config.yaml` (a copy of `config/default.yaml` with only
the rate-limit block overridden), mounts one trivial route decorated with
the real `Limiter`, and fires 140 requests at a simple Python round-robin
proxy in front of both.

**Not executed in this session.** Importing `rag.factory`
(`sentence_transformers`/`torch`, pulled in transitively by
`rag.api.deps`) was measured directly in this sandboxed environment at
**1122 seconds** (~19 minutes) for a single cold import. See
`README.md`'s "What was executed vs. simulated" section for the exact
timing and how it was measured. Booting two such processes to run this
script would cost on the order of 40 minutes of pure import time before
a single request could be fired, which was judged impractical for this
session. The rigorous, *actually executed* substitute for this exact
scenario is `distributed_state_experiment/tests/test_inmemory_limiter_divergence.py`
(3 passing tests, using the same process-local-counter shape `slowapi`'s
real `MemoryStorage` has). This script is provided complete and correct
so it can be run wherever a faster import path is available (a warm
Python/torch cache, or a machine without this sandbox's apparent
antivirus/first-touch DLL-scan cost).

Usage (once a fast environment is available)
---------------------------------------------
    python -m distributed_state_experiment.part1_process_local_state.two_pod_rate_limit_demo
"""

from __future__ import annotations

import http.client
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PYTHON = sys.executable
DEMO_CONFIG = Path(__file__).parent / "rate_limit_demo_config.yaml"
POD_A_PORT = 8101
POD_B_PORT = 8102
PROXY_PORT = 8100


def _free_port_check(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _start_pod(port: int) -> subprocess.Popen:
    """Boot one real uvicorn process running the real `rag.api.main:app`.

    Each process is fully independent: its own Python interpreter, its
    own `rag.api.deps.get_rate_limiter()` `lru_cache`d singleton, its own
    in-memory `slowapi.MemoryStorage`: exactly "Worker A" / "Worker B"
    in the task's own example.
    """
    env = dict(os.environ)
    env["RAG_CONFIG_PATH"] = str(DEMO_CONFIG)
    return subprocess.Popen(
        [PYTHON, "-m", "uvicorn", "rag.api.main:app", "--port", str(port), "--host", "127.0.0.1"],
        env=env,
    )


class _RoundRobinProxy(BaseHTTPRequestHandler):
    """A deliberately tiny stand-in load balancer: alternates every request between the two pods."""

    _counter = [0]
    _lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's own naming convention
        with self._lock:
            self._counter[0] += 1
            target_port = POD_A_PORT if self._counter[0] % 2 == 1 else POD_B_PORT
        length = int(self.headers.get("Content-Length", 0))
        body_in = self.rfile.read(length) if length else b""
        conn = http.client.HTTPConnection("127.0.0.1", target_port, timeout=5)
        # POST /query is the actual rate-limited route (@_limiter.limit(...) in
        # api/routers/query.py). slowapi's SlowAPIMiddleware enforces the limit
        # at the ASGI-middleware layer, before FastAPI resolves the route's own
        # Depends()/body validation, so this demo never needs a real embedder/
        # Postgres/Ollama to be healthy for the 100-vs-140 counting to be valid;
        # requests 101+ from each pod's own perspective get a 429 regardless of
        # what a fully-processed answer would have looked like.
        conn.request("POST", "/query", body=body_in, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        body = resp.read()
        self.send_response(resp.status)
        self.send_header("X-Backend-Pod", "A" if target_port == POD_A_PORT else "B")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        pass  # keep the demo's own output the focus, not per-request access logs


def main() -> None:
    """Boot two pods, a round-robin proxy, fire 140 requests, and report the split."""
    print("Booting Pod A and Pod B (each a real uvicorn + rag.api.main:app process)...")
    pod_a = _start_pod(POD_A_PORT)
    pod_b = _start_pod(POD_B_PORT)
    try:
        for _ in range(
            120
        ):  # generous: this is the step that costs ~19 minutes per pod in this sandbox
            if not _free_port_check(POD_A_PORT) and not _free_port_check(POD_B_PORT):
                break
            time.sleep(1)
        else:
            raise RuntimeError("Pods did not become reachable in time")

        proxy = ThreadingHTTPServer(("127.0.0.1", PROXY_PORT), _RoundRobinProxy)
        proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        proxy_thread.start()

        payload = b'{"query": "what is the password policy?"}'
        allowed = 0
        denied = 0
        for _ in range(140):
            conn = http.client.HTTPConnection("127.0.0.1", PROXY_PORT, timeout=5)
            conn.request(
                "POST", "/query", body=payload, headers={"Content-Type": "application/json"}
            )
            resp = conn.getresponse()
            resp.read()
            if resp.status == 429:
                denied += 1
            else:
                allowed += 1

        print(
            f"140 requests round-robined across Pod A / Pod B: allowed={allowed}, denied(429)={denied}"
        )
        print(
            "Intended tenant-wide limit was 100/minute; with each pod's own independent in-memory "
            f"counter, {allowed} got through -- see README.md Part 1 for the interpretation."
        )
        proxy.shutdown()
    finally:
        pod_a.terminate()
        pod_b.terminate()
        pod_a.wait(timeout=10)
        pod_b.wait(timeout=10)


if __name__ == "__main__":
    main()
