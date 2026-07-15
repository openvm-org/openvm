| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3021/fibonacci-b23223620033a1df48bbd1eaf33e8bcbe2b113bd.md) | 414 |  4,000,051 |  226 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3021/keccak-b23223620033a1df48bbd1eaf33e8bcbe2b113bd.md) | 8,375 |  14,365,133 |  1,510 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3021/sha2_bench-b23223620033a1df48bbd1eaf33e8bcbe2b113bd.md) | 3,965 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3021/regex-b23223620033a1df48bbd1eaf33e8bcbe2b113bd.md) | 571 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3021/ecrecover-b23223620033a1df48bbd1eaf33e8bcbe2b113bd.md) | 219 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3021/pairing-b23223620033a1df48bbd1eaf33e8bcbe2b113bd.md) | 265 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3021/kitchen_sink-b23223620033a1df48bbd1eaf33e8bcbe2b113bd.md) | 1,908 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b23223620033a1df48bbd1eaf33e8bcbe2b113bd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29435096446)
