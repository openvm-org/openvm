| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/fibonacci-c0561903201f6b10dd31a454154e2e74cf941400.md) | 1,020 |  4,000,051 |  388 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/keccak-c0561903201f6b10dd31a454154e2e74cf941400.md) | 16,356 |  14,365,133 |  3,033 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/sha2_bench-c0561903201f6b10dd31a454154e2e74cf941400.md) | 8,198 |  11,167,961 |  999 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/regex-c0561903201f6b10dd31a454154e2e74cf941400.md) | 1,214 |  4,090,656 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/ecrecover-c0561903201f6b10dd31a454154e2e74cf941400.md) | 436 |  112,210 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/pairing-c0561903201f6b10dd31a454154e2e74cf941400.md) | 595 |  592,827 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/kitchen_sink-c0561903201f6b10dd31a454154e2e74cf941400.md) | 3,966 |  1,979,971 |  876 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c0561903201f6b10dd31a454154e2e74cf941400

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27821446918)
