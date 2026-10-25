# Context Engineering benchmark

Generated at: 2026-10-25T09:56:24.543017+00:00
Total scenario/strategy/size combinations: 240

| Strategy | Mean utilization | Mean reduction | Mean build (ms) | Dropped items | Compressed items | Runs |
|---|---|---|---|---|---|---|
| A: top-N (baseline) | 0.136 | 0.244 | 0.562 | 1861 | 0 | 60 |
| B: naive token-first | 0.225 | 0.196 | 0.550 | 1348 | 0 | 60 |
| C: token-budget + dedup | 0.043 | 0.342 | 0.892 | 5 | 0 | 60 |
| D: full pipeline | 0.031 | 0.357 | 1.160 | 0 | 6 | 60 |

Candidate-size sweep: 2000, 8000, 16000, 32000 tokens.
