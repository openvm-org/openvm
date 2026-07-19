| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/fibonacci-d0008799260d102013cc794fae8fba1c400f4998.md) | 413 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/keccak-d0008799260d102013cc794fae8fba1c400f4998.md) | 8,652 |  14,365,133 |  1,525 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/sha2_bench-d0008799260d102013cc794fae8fba1c400f4998.md) | 4,187 |  11,167,961 |  519 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/regex-d0008799260d102013cc794fae8fba1c400f4998.md) | 566 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/ecrecover-d0008799260d102013cc794fae8fba1c400f4998.md) | 220 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/pairing-d0008799260d102013cc794fae8fba1c400f4998.md) | 294 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/kitchen_sink-d0008799260d102013cc794fae8fba1c400f4998.md) | 1,917 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d0008799260d102013cc794fae8fba1c400f4998

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29700989596)
