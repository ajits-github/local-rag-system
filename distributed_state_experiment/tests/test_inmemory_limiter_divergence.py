"""Part 1: prove two process-local limiter instances diverge under identical traffic.

No Redis, no HTTP server, no `rag` import needed -- this is the
"analytical demonstration" half of Part 1 (see
`part1_process_local_state/README_PART1.md` / the main README's Part 1
section for the executed-two-OS-processes half). It directly instantiates
two independent `InMemoryFixedWindowLimiter`s (the same process-local
shape `slowapi.MemoryStorage` has) to model "Pod A" and "Pod B" sharing
one intended limit but each counting in its own memory.
"""

from __future__ import annotations

from distributed_state_experiment.rate_limiter.in_memory import InMemoryFixedWindowLimiter


def test_single_instance_enforces_its_own_limit() -> None:
    """A single limiter (the single-replica case) correctly caps at the limit."""
    limiter = InMemoryFixedWindowLimiter(limit=100, window_seconds=60, pod_name="only-pod")
    allowed_count = sum(1 for _ in range(140) if limiter.hit("tenant:acme")[0])
    assert allowed_count == 100


def test_two_independent_instances_double_the_effective_limit() -> None:
    """Two pods, one intended tenant limit of 100/minute, load-balanced 70/70 -> 140 allowed globally.

    This is the exact scenario the task's own example describes: each pod
    independently allows up to 100 before refusing, so splitting 140
    requests 70/70 between them lets all 140 through even though the
    tenant's intended global limit was 100.
    """
    pod_a = InMemoryFixedWindowLimiter(limit=100, window_seconds=60, pod_name="pod-a")
    pod_b = InMemoryFixedWindowLimiter(limit=100, window_seconds=60, pod_name="pod-b")

    allowed_a = sum(1 for _ in range(70) if pod_a.hit("tenant:acme")[0])
    allowed_b = sum(1 for _ in range(70) if pod_b.hit("tenant:acme")[0])

    assert allowed_a == 70
    assert allowed_b == 70
    global_allowed = allowed_a + allowed_b
    assert global_allowed == 140
    assert global_allowed > 100, "the intended tenant-wide limit was exceeded across replicas"


def test_round_robin_load_balancing_still_double_counts() -> None:
    """Even perfectly-alternated round-robin traffic still exceeds the shared intended limit.

    Demonstrates the bug is not about *how* the load balancer distributes
    traffic (round-robin vs. random vs. skewed) -- any split across N
    independent in-memory counters raises the effective limit toward
    N x the intended per-replica limit, simply because there is no shared
    state to intersect against.
    """
    pods = [
        InMemoryFixedWindowLimiter(limit=100, window_seconds=60, pod_name=f"pod-{i}")
        for i in range(3)
    ]
    total_requests = 240
    total_allowed = 0
    for i in range(total_requests):
        pod = pods[i % len(pods)]
        allowed, _count = pod.hit("tenant:acme")
        total_allowed += int(allowed)

    # 240 requests round-robined across 3 pods = 80 each, all under each
    # pod's own 100/min cap, so every single one is allowed -- 240 > 100.
    assert total_allowed == 240
    assert total_allowed > 100
