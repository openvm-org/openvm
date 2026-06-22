| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/fibonacci-3b89ec672c53d59168bede30ad3a98484467d6d7.md) | 1,038 |  4,000,051 |  397 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/keccak-3b89ec672c53d59168bede30ad3a98484467d6d7.md) | 16,324 |  14,365,133 |  3,021 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/sha2_bench-3b89ec672c53d59168bede30ad3a98484467d6d7.md) | 8,291 |  11,167,961 |  1,005 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/regex-3b89ec672c53d59168bede30ad3a98484467d6d7.md) | 1,244 |  4,090,656 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/ecrecover-3b89ec672c53d59168bede30ad3a98484467d6d7.md) | 436 |  112,210 |  279 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/pairing-3b89ec672c53d59168bede30ad3a98484467d6d7.md) | 599 |  592,827 |  299 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/kitchen_sink-3b89ec672c53d59168bede30ad3a98484467d6d7.md) | 3,891 |  1,979,971 |  864 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3b89ec672c53d59168bede30ad3a98484467d6d7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27945385808)
