# Context Engineering benchmark

Generated at: 2026-11-02T10:17:22.617672+00:00
Total scenario/strategy/size combinations: 240

| Strategy | Mean utilization | Overflowing runs | Mean excess tokens | Mean reduction | Mean build (ms) | Dropped items | Compressed items | Runs |
|---|---|---|---|---|---|---|---|---|
| A: top-N (baseline) | 0.333 | 5/60 | 591.0 | 0.244 | 2.609 | 1861 | 0 | 60 |
| B: naive token-first | 0.244 | 0/60 | 0.0 | 0.206 | 2.525 | 1425 | 0 | 60 |
| C: token-budget + dedup | 0.051 | 0/60 | 0.0 | 0.342 | 5.368 | 5 | 0 | 60 |
| D: full pipeline | 0.038 | 0/60 | 0.0 | 0.358 | 6.807 | 0 | 9 | 60 |

Candidate-size sweep: 2000, 8000, 16000, 32000 tokens.
