| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/fibonacci-307b6266171c2f1af19bb489eab0e7b017c75517.md) | 407 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/keccak-307b6266171c2f1af19bb489eab0e7b017c75517.md) | 8,579 |  14,365,133 |  1,517 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/sha2_bench-307b6266171c2f1af19bb489eab0e7b017c75517.md) | 4,234 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/regex-307b6266171c2f1af19bb489eab0e7b017c75517.md) | 570 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/ecrecover-307b6266171c2f1af19bb489eab0e7b017c75517.md) | 219 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/pairing-307b6266171c2f1af19bb489eab0e7b017c75517.md) | 285 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/kitchen_sink-307b6266171c2f1af19bb489eab0e7b017c75517.md) | 1,922 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/307b6266171c2f1af19bb489eab0e7b017c75517

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29701398366)
