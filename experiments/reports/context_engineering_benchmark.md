# Context Engineering benchmark

Generated at: 2026-10-29T13:50:24.285454+00:00
Total scenario/strategy/size combinations: 240

| Strategy | Mean utilization | Overflowing runs | Mean excess tokens | Mean reduction | Mean build (ms) | Dropped items | Compressed items | Runs |
|---|---|---|---|---|---|---|---|---|
| A: top-N (baseline) | 0.322 | 5/60 | 590.0 | 0.244 | 0.773 | 1861 | 0 | 60 |
| B: naive token-first | 0.225 | 0/60 | 0.0 | 0.196 | 0.903 | 1348 | 0 | 60 |
| C: token-budget + dedup | 0.043 | 0/60 | 0.0 | 0.342 | 1.372 | 5 | 0 | 60 |
| D: full pipeline | 0.031 | 0/60 | 0.0 | 0.357 | 1.758 | 0 | 6 | 60 |

Candidate-size sweep: 2000, 8000, 16000, 32000 tokens.
