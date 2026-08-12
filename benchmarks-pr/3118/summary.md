| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3118/fibonacci-98b066ab14dec56f1cdbf1694d2c30661d318439.md) | 462 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3118/keccak-98b066ab14dec56f1cdbf1694d2c30661d318439.md) | 7,369 |  14,365,133 |  1,529 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3118/sha2_bench-98b066ab14dec56f1cdbf1694d2c30661d318439.md) | 4,203 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3118/regex-98b066ab14dec56f1cdbf1694d2c30661d318439.md) | 667 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3118/ecrecover-98b066ab14dec56f1cdbf1694d2c30661d318439.md) | 200 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3118/pairing-98b066ab14dec56f1cdbf1694d2c30661d318439.md) | 232 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3118/kitchen_sink-98b066ab14dec56f1cdbf1694d2c30661d318439.md) | 2,048 |  1,979,971 |  464 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/98b066ab14dec56f1cdbf1694d2c30661d318439

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31621351074)
