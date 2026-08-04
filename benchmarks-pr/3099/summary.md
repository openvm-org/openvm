| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/fibonacci-1d4d0662925b3a6bdde508a6205b0664a414c8b9.md) | 1,573 |  12,000,265 |  360 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/keccak-1d4d0662925b3a6bdde508a6205b0664a414c8b9.md) | 9,339 |  18,655,329 |  1,528 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/sha2_bench-1d4d0662925b3a6bdde508a6205b0664a414c8b9.md) | 4,971 |  14,793,960 |  576 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/regex-1d4d0662925b3a6bdde508a6205b0664a414c8b9.md) | 661 |  4,137,067 |  209 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/ecrecover-1d4d0662925b3a6bdde508a6205b0664a414c8b9.md) | 435 |  123,583 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/pairing-1d4d0662925b3a6bdde508a6205b0664a414c8b9.md) | 595 |  1,745,757 |  192 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/kitchen_sink-1d4d0662925b3a6bdde508a6205b0664a414c8b9.md) | 2,202 |  2,579,903 |  478 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1d4d0662925b3a6bdde508a6205b0664a414c8b9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30929354964)
