| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2891/fibonacci-a8e49a1fce2e6efd69fed08f02c16175e799be82.md) | 1,646 |  4,000,051 |  523 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2891/keccak-a8e49a1fce2e6efd69fed08f02c16175e799be82.md) | 16,273 |  14,365,133 |  3,039 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2891/sha2_bench-a8e49a1fce2e6efd69fed08f02c16175e799be82.md) | 10,404 |  11,167,961 |  1,924 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2891/regex-a8e49a1fce2e6efd69fed08f02c16175e799be82.md) | 1,537 |  4,090,656 |  432 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2891/ecrecover-a8e49a1fce2e6efd69fed08f02c16175e799be82.md) | 481 |  112,210 |  312 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2891/pairing-a8e49a1fce2e6efd69fed08f02c16175e799be82.md) | 623 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2891/kitchen_sink-a8e49a1fce2e6efd69fed08f02c16175e799be82.md) | 3,970 |  1,979,971 |  868 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a8e49a1fce2e6efd69fed08f02c16175e799be82

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27567208649)
