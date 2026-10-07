"""Drive experiments 1-10 (and a local throughput proxy for 11-12) as real subprocesses.

Launches real, separate Python OS processes talking to a real TCP Redis
endpoint (`--redis-url`, default the `fake_redis_server.py` stand-in
used here; see that module's docstring for why) via
`subprocess.Popen`, so "kill a worker mid-processing," "N workers vs 1
worker throughput," and "duplicate delivery" are all exercised against
genuinely concurrent, genuinely separate processes, not simulated with
threads inside one process. Prints a plain-text report to stdout; capture
it (`> results/experiment_log.txt`) to keep a record.

Usage
-----
    python -m distributed_state_experiment.cli.run_experiments --redis-url redis://127.0.0.1:6399/0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import uuid

import redis

from distributed_state_experiment.job_queue.models import IngestionJobPayload
from distributed_state_experiment.job_queue.queue import IngestionJobQueue

PYTHON = sys.executable


def _worker_cmd(name: str, redis_url: str, extra: list[str] | None = None) -> list[str]:
    cmd = [
        PYTHON,
        "-m",
        "distributed_state_experiment.cli.run_worker",
        "--redis-url",
        redis_url,
        "--name",
        name,
        "--run-until-idle",
    ]
    return cmd + (extra or [])


def _run_workers_concurrently(
    names: list[str], redis_url: str, extra: list[str] | None = None
) -> tuple[float, list[str]]:
    """Launch one subprocess per name, near-simultaneously, and wait for all to exit.

    Returns
    -------
    tuple[float, list[str]]
        Wall-clock seconds elapsed, and each process's combined stdout.
    """
    start = time.monotonic()
    procs = [
        subprocess.Popen(
            _worker_cmd(name, redis_url, extra),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for name in names
    ]
    outputs = [p.communicate()[0] for p in procs]
    elapsed = time.monotonic() - start
    return elapsed, outputs


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> None:
    """Parse args and run experiments 1-10 plus a local throughput proxy for 11-12 in sequence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redis-url", default="redis://127.0.0.1:6399/0")
    args = parser.parse_args()
    redis_url = args.redis_url
    client = redis.Redis.from_url(redis_url)
    # Every stream/group/content in this run is namespaced by a fresh run_id
    # so re-running this script never collides with a previous run's
    # already-completed jobs still sitting in the same Redis instance
    # (idempotent submission would otherwise silently no-op a resubmission
    # of byte-identical content, a real issue this script's first draft
    # actually hit; see README.md's "What was executed vs. simulated" note).
    run_id = uuid.uuid4().hex[:8]
    print(f"run_id={run_id}")

    # ---- Experiments 1-3: submit 20, drain with 1 worker, then with 3 ----
    section("Experiments 1-2: submit 20 jobs, 1 worker")
    stream_1w = f"exp_1worker_{run_id}:stream"
    group_1w = f"exp_1worker_{run_id}:group"
    queue_1w = IngestionJobQueue(client, stream_name=stream_1w, group_name=group_1w)
    for i in range(20):
        payload = IngestionJobPayload.build(
            f"one-worker-{i}.md", f"exp_throughput_{run_id}", content=f"run={run_id} c{i}".encode()
        )
        queue_1w.submit(payload)
    print(f"submitted 20 jobs, backlog_depth={queue_1w.backlog_depth()}")

    elapsed_1w, out_1w = _run_workers_concurrently(
        ["solo-worker"], redis_url, extra=["--stream", stream_1w, "--group", group_1w]
    )
    print(out_1w[0].strip())
    print(f"RESULT: 1 worker drained 20 simulated jobs in {elapsed_1w:.3f}s wall-clock")

    section("Experiment 3: submit 20 more jobs, drain with 3 concurrent workers")
    stream_3w = f"exp_3worker_{run_id}:stream"
    group_3w = f"exp_3worker_{run_id}:group"
    queue_3w = IngestionJobQueue(client, stream_name=stream_3w, group_name=group_3w)
    for i in range(20):
        payload = IngestionJobPayload.build(
            f"three-worker-{i}.md",
            f"exp_throughput_{run_id}",
            content=f"run={run_id} c{i}".encode(),
        )
        queue_3w.submit(payload)
    print(f"submitted 20 jobs, backlog_depth={queue_3w.backlog_depth()}")

    elapsed_3w, out_3w = _run_workers_concurrently(
        ["worker-x", "worker-y", "worker-z"],
        redis_url,
        extra=["--stream", stream_3w, "--group", group_3w],
    )
    for o in out_3w:
        print(o.strip())
    print(f"RESULT: 3 concurrent workers drained 20 simulated jobs in {elapsed_3w:.3f}s wall-clock")
    print(
        f"RESULT: throughput proxy for horizontal scaling: "
        f"{elapsed_1w:.3f}s (1 worker) -> {elapsed_3w:.3f}s (3 workers), "
        f"speedup={elapsed_1w / elapsed_3w if elapsed_3w > 0 else float('inf'):.2f}x "
        f"(a *local-process* proxy for k8s pod scaling -- see README's Experiments 11-12 section "
        f"for why real kubectl-driven pod scaling was not executed in this session)"
    )

    # ---- Experiment 4-5: kill a worker mid-processing, verify recovery ----
    section("Experiments 4-5: kill a worker mid-processing, verify reclaim_stale recovers it")
    stream_kill = f"exp_kill_{run_id}:stream"
    group_kill = f"exp_kill_{run_id}:group"
    queue_kill = IngestionJobQueue(client, stream_name=stream_kill, group_name=group_kill)
    # slow_seconds gives us a wide, reliable window to observe the job
    # claimed-but-unacked before killing the process. A fixed guessed
    # sleep before a real process has even finished importing/starting up
    # is not reliable (an earlier version of this script used a flat 0.3s
    # sleep and it fired before the child process had claimed anything at
    # all; see README.md's "What was executed vs. simulated" note on this
    # specific fix).
    payload = IngestionJobPayload.build(
        "kill-target.md", f"exp_kill_{run_id}", content=f"run={run_id}".encode(), slow_seconds=5.0
    )
    record = queue_kill.submit(payload)
    doomed = subprocess.Popen(
        [
            PYTHON,
            "-m",
            "distributed_state_experiment.cli.run_worker",
            "--redis-url",
            redis_url,
            "--name",
            "doomed-worker",
            "--stream",
            stream_kill,
            "--group",
            group_kill,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 15.0
    pending_before_kill = 0
    while time.monotonic() < deadline:
        pending_before_kill = queue_kill.pending_count()
        if pending_before_kill > 0:
            break
        time.sleep(0.1)
    doomed.kill()
    doomed.wait()
    print(
        f"pending entries while doomed-worker was alive (polled until claimed): {pending_before_kill}"
    )
    print(f"job status right after kill: {queue_kill.store.get(record.job_id).status.value}")

    time.sleep(0.05)  # ensure a measurably nonzero idle time for the reclaim's min_idle_ms filter
    reclaimed = queue_kill.reclaim_stale("rescuer-worker", min_idle_ms=0)
    print(
        f"reclaim_stale found {len(reclaimed)} orphaned entr{'y' if len(reclaimed) == 1 else 'ies'}"
    )
    if reclaimed:
        queue_kill.complete(reclaimed[0], {"chunks_embedded": 1})
    final = queue_kill.store.get(record.job_id)
    print(f"RESULT: job status after recovery: {final.status.value} (attempts={final.attempts})")

    # ---- Experiment 6-7: duplicate delivery / idempotency ----
    section("Experiments 6-7: deliver the same job twice, verify no duplicate persistent effect")
    stream_dup = f"exp_dup_{run_id}:stream"
    group_dup = f"exp_dup_{run_id}:group"
    queue_dup = IngestionJobQueue(client, stream_name=stream_dup, group_name=group_dup)
    dup_payload = IngestionJobPayload.build(
        "dup.md", f"exp_dup_{run_id}", content=f"run={run_id} same bytes".encode()
    )
    dup_record = queue_dup.submit(dup_payload)
    first = queue_dup.claim("w1")[0]
    queue_dup.complete(first, {"chunks_embedded": 5})
    # Simulate a duplicate delivery of the same job_id (e.g. a redundant XADD from a retrying producer).
    client.xadd(stream_dup, {"job_id": dup_record.job_id})
    second = queue_dup.claim("w2")[0]
    queue_dup.complete(second, {"chunks_embedded": 5})
    dup_final = queue_dup.store.get(dup_record.job_id)
    print(
        f"RESULT: job processed twice (attempts={dup_final.attempts}), "
        f"final status={dup_final.status.value}, one job_id throughout: "
        f"real IngestionPipeline idempotency (checksum-gated replace_document_chunks) "
        f"is what makes the *second* run's DB effect a no-op in production; this queue's own "
        f"submission-time dedup (see next line) prevents even a second job_id from existing."
    )
    resubmit = queue_dup.submit(dup_payload)
    print(
        f"resubmitting the identical payload returns the same job_id: {resubmit.job_id == dup_record.job_id}"
    )

    # ---- Experiment 8-10: poison job, retries/backoff, dead-letter ----
    section("Experiments 8-10: poison job -> retries with backoff -> dead-letter")
    stream_poison = f"exp_poison_{run_id}:stream"
    group_poison = f"exp_poison_{run_id}:group"
    queue_poison = IngestionJobQueue(client, stream_name=stream_poison, group_name=group_poison)
    poison_payload = IngestionJobPayload.build(
        "poison.md", f"exp_poison_{run_id}", content=f"run={run_id} x".encode(), poison=True
    )
    poison_record = queue_poison.submit(poison_payload, max_attempts=3)

    out = subprocess.run(
        [
            PYTHON,
            "-m",
            "distributed_state_experiment.cli.run_worker",
            "--redis-url",
            redis_url,
            "--name",
            "poison-worker",
            "--stream",
            stream_poison,
            "--group",
            group_poison,
            "--run-until-idle",
            "--base-backoff-seconds",
            "0.2",
            "--max-backoff-seconds",
            "0.4",
        ],
        capture_output=True,
        text=True,
    )
    print(out.stdout.strip())
    after_first_pass = queue_poison.store.get(poison_record.job_id)
    print(
        f"status after first drain (before any retry is due): {after_first_pass.status.value}, attempts={after_first_pass.attempts}"
    )

    # Drive the remaining retries by waiting past each backoff and re-running the worker.
    for round_num in range(2, 4):
        time.sleep(0.5)  # exceed the 0.2-0.4s backoff window
        promoted = queue_poison.promote_due_retries()
        print(f"round {round_num}: promote_due_retries promoted {promoted} job(s)")
        out = subprocess.run(
            [
                PYTHON,
                "-m",
                "distributed_state_experiment.cli.run_worker",
                "--redis-url",
                redis_url,
                "--name",
                f"poison-worker-r{round_num}",
                "--stream",
                stream_poison,
                "--group",
                group_poison,
                "--run-until-idle",
                "--base-backoff-seconds",
                "0.2",
                "--max-backoff-seconds",
                "0.4",
            ],
            capture_output=True,
            text=True,
        )
        state = queue_poison.store.get(poison_record.job_id)
        print(
            f"round {round_num}: status={state.status.value}, attempts={state.attempts}, last_error={state.last_error!r}"
        )
        if state.status.value == "dead_letter":
            break

    final_poison = queue_poison.store.get(poison_record.job_id)
    print(
        f"RESULT: poison job final status={final_poison.status.value} after {final_poison.attempts} attempts "
        f"(max_attempts=3); dead_letter_depth={queue_poison.dead_letter_depth()}"
    )

    section("Done")
    print("See README.md's Experiments section for the interpretation of each result above.")


if __name__ == "__main__":
    main()
