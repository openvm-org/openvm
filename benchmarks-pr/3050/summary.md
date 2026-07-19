| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/fibonacci-2e9ee7b221f73c891d5dbe499828b96413ebc418.md) | 413 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/keccak-2e9ee7b221f73c891d5dbe499828b96413ebc418.md) | 8,595 |  14,365,133 |  1,521 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/sha2_bench-2e9ee7b221f73c891d5dbe499828b96413ebc418.md) | 4,236 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/regex-2e9ee7b221f73c891d5dbe499828b96413ebc418.md) | 571 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/ecrecover-2e9ee7b221f73c891d5dbe499828b96413ebc418.md) | 216 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/pairing-2e9ee7b221f73c891d5dbe499828b96413ebc418.md) | 294 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/kitchen_sink-2e9ee7b221f73c891d5dbe499828b96413ebc418.md) | 1,912 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2e9ee7b221f73c891d5dbe499828b96413ebc418

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29702544631)
