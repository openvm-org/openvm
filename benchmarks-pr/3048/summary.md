| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/fibonacci-7f9d02adbe1f40bc21cc608849295d61a0089318.md) | 469 |  4,000,051 |  239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/keccak-7f9d02adbe1f40bc21cc608849295d61a0089318.md) | 7,338 |  14,365,133 |  1,534 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/sha2_bench-7f9d02adbe1f40bc21cc608849295d61a0089318.md) | 4,709 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/regex-7f9d02adbe1f40bc21cc608849295d61a0089318.md) | 671 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/ecrecover-7f9d02adbe1f40bc21cc608849295d61a0089318.md) | 228 |  112,210 |  189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/pairing-7f9d02adbe1f40bc21cc608849295d61a0089318.md) | 324 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/kitchen_sink-7f9d02adbe1f40bc21cc608849295d61a0089318.md) | 2,693 |  1,979,971 |  467 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7f9d02adbe1f40bc21cc608849295d61a0089318

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29953524993)
