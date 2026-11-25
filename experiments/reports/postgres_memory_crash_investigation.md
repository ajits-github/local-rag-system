# PostgreSQL crash under the freshness scale benchmark

Summary of the evidence behind **OPS-019** in `KNOWN_ISSUES.md`. The raw
cgroup sample log is not tracked; every figure below was taken from it, from
the PostgreSQL server log, or from the tracked result files listed at the end.

## Setup

- `local-rag-postgres` container: PostgreSQL 16.14, cgroup `memory.max`
  1 GiB (the dev compose limit), `shared_buffers` 128MB, `work_mem` 4MB,
  Docker Desktop on WSL2.
- Workload: the database part of `scripts/benchmark_freshness.py` with
  synthetic `bench_fresh_*` tables, 3 chunks per document, up to 100,000
  documents (300,000 chunk rows, a heap of about 1 GB). It bulk-inserts each
  size, then runs version fetches and dense retrievals.
- Crash signature every time: `server process (PID n) exited with exit
  code 2`, no preceding `ERROR`, then `terminating any other active server
  processes` and automatic recovery.

## Crashes observed

| When (UTC) | Run | Memory limit |
|---|---|---|
| 2026-11-22 18:26:22 | first scale run, database part (query not logged) | 1 GiB |
| 2026-11-22 19:05:16 | scale run at 50,000 documents, after the `chunk_index = 0` fetch completed | 1 GiB |
| 2026-11-22 19:17:11 | second old-style `DISTINCT` fetch on 100,000 documents, in `psql` | 1 GiB |
| 2026-11-23 13:54:53 | controlled rerun of the full workload, with memory sampling | 1 GiB |

No crash occurred in the 2 GiB rerun described below.

## Memory measurements (controlled rerun, 2026-11-23)

The container's `memory.current`, swap use, and `memory.events` counters
were sampled every 2 to 3 seconds from 13:30 to 14:22 UTC (1,267 samples).

| Phase (UTC) | Memory median / max | Samples at or above 1,000 MiB | Peak swap | `memory.events` `max` counter | `oom_kill` |
|---|---|---|---|---|---|
| 13:30 to 13:39, idle before load | 108 / 1,014 MiB | 1 of 226 | 3 MiB | 0 | 0 |
| 13:39 to 13:54, full workload at 1 GiB | 1,024 / 1,024 MiB | 339 of 357 | 60 MiB | 34 to 18,471 | 0 |
| 13:54:53, crash | 1,024 MiB | | 62.6 MiB (the trace's peak, in the same sample) | 19,356 | 0 |
| 13:56 to 13:58, old `DISTINCT` alone, 4 runs at 1 GiB | 1,024 / 1,024 MiB | 47 of 57 | 15 MiB | 19,841 to 25,366 | 0 |
| 13:59 to 14:22, full workload with the limit raised to 2 GiB | 1,448 / 1,644 MiB | not applicable | 13 MiB | unchanged at 25,366 | 0 |

- At 1 GiB the container sat at its limit for the whole workload, and the
  crash came at the peak of swap use.
- The four standalone `DISTINCT` runs at 1 GiB all completed (26 to 37 s
  each) and added about 5,500 limit hits, with no crash.
- With 2 GiB (`docker update`, reverted afterwards) the same workload
  completed with no crash and no new limit hits. Peak usage was 1,644 MiB
  (1.72 GB).
- `oom_kill` stayed 0 throughout. A kernel OOM kill would appear in the
  server log as `terminated by signal 9`; exit code 2 is PostgreSQL's
  quick-exit (SIGQUIT) path.

## Conclusion

The crashes follow from running PostgreSQL pinned at a 1 GiB limit under a
workload that needs about 1.7 GB: constant reclaim and some swapping, then a
backend exits through the quick-exit path. The backend that exited was
short-lived and logged nothing, and the kernel log from the earlier crashes
was lost when the Docker VM restarted.

## Limitations

- Small sample. At 1 GiB, all 3 full-workload runs crashed, and so did 1
  standalone `DISTINCT` in `psql` on 2026-11-22, run on tables a crashed
  benchmark had just loaded; 4 standalone `DISTINCT` runs on 2026-11-23
  completed. At 2 GiB, the 1 full-workload run completed.
- What delivered the quit signal is not proven. The evidence rules out the
  kernel OOM killer for the 2026-11-23 crash only; no kernel log exists for
  the 2026-11-22 crashes.
- The trace covers one machine (Docker Desktop on WSL2, 8 logical CPUs) and
  synthetic data. It says nothing about production sizing beyond "more than
  1 GiB for about 100,000 documents with this schema".
- The 1 GiB dev compose limit is unchanged. Raising it is a deployment
  decision, not part of the freshness change.

## Tracked sources

- `experiments/results/freshness/scale_db_fetch_psql_50k_100k.json`: the
  2026-11-22 crashes and fetch timings at 1 GiB.
- `experiments/results/freshness/scale_db_2g_37_1_in_container.json`: the
  2 GiB rerun's retrieval timings.
- `experiments/reports/freshness_benchmark.md`: the full benchmark report.
