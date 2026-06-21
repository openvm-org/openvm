| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/fibonacci-d88993affb6fcd247cf7df6a1b8bf35b12dc8974.md) | 1,026 |  4,000,051 |  387 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/keccak-d88993affb6fcd247cf7df6a1b8bf35b12dc8974.md) | 16,338 |  14,365,133 |  3,061 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/sha2_bench-d88993affb6fcd247cf7df6a1b8bf35b12dc8974.md) | 8,181 |  11,167,961 |  1,005 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/regex-d88993affb6fcd247cf7df6a1b8bf35b12dc8974.md) | 1,192 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/ecrecover-d88993affb6fcd247cf7df6a1b8bf35b12dc8974.md) | 437 |  112,210 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/pairing-d88993affb6fcd247cf7df6a1b8bf35b12dc8974.md) | 596 |  592,827 |  293 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/kitchen_sink-d88993affb6fcd247cf7df6a1b8bf35b12dc8974.md) | 3,891 |  1,979,971 |  860 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d88993affb6fcd247cf7df6a1b8bf35b12dc8974

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27909940344)
