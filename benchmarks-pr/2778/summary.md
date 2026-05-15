| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-ebfb12e2465f1dea9e46b0a75d9dc7311e1b82c9.md) | 1,413 |  4,000,051 |  433 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-ebfb12e2465f1dea9e46b0a75d9dc7311e1b82c9.md) | 13,237 |  14,365,133 |  2,192 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-ebfb12e2465f1dea9e46b0a75d9dc7311e1b82c9.md) | 8,998 |  11,167,961 |  1,408 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-ebfb12e2465f1dea9e46b0a75d9dc7311e1b82c9.md) | 1,336 |  4,090,656 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-ebfb12e2465f1dea9e46b0a75d9dc7311e1b82c9.md) | 469 |  112,210 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-ebfb12e2465f1dea9e46b0a75d9dc7311e1b82c9.md) | 593 |  592,827 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-ebfb12e2465f1dea9e46b0a75d9dc7311e1b82c9.md) | 1,787 |  1,979,971 |  407 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ebfb12e2465f1dea9e46b0a75d9dc7311e1b82c9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25909007796)
