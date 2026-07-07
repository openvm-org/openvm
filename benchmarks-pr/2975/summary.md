| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2975/fibonacci-009a6523b333e53117778248878275f46b516eb8.md) | 869 |  4,000,051 |  396 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2975/keccak-009a6523b333e53117778248878275f46b516eb8.md) | 15,309 |  14,365,133 |  3,014 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2975/sha2_bench-009a6523b333e53117778248878275f46b516eb8.md) | 7,971 |  11,167,961 |  999 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2975/regex-009a6523b333e53117778248878275f46b516eb8.md) | 1,027 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2975/ecrecover-009a6523b333e53117778248878275f46b516eb8.md) | 308 |  112,210 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2975/pairing-009a6523b333e53117778248878275f46b516eb8.md) | 452 |  592,827 |  295 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2975/kitchen_sink-009a6523b333e53117778248878275f46b516eb8.md) | 3,715 |  1,979,971 |  853 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/009a6523b333e53117778248878275f46b516eb8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28898382258)
