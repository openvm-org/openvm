| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3094/fibonacci-08cb81327fcb2c5e8bcba94e615183d67828c526.md) | 478 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3094/keccak-08cb81327fcb2c5e8bcba94e615183d67828c526.md) | 7,331 |  14,365,133 |  1,519 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3094/sha2_bench-08cb81327fcb2c5e8bcba94e615183d67828c526.md) | 4,134 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3094/regex-08cb81327fcb2c5e8bcba94e615183d67828c526.md) | 658 |  4,090,656 |  211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3094/ecrecover-08cb81327fcb2c5e8bcba94e615183d67828c526.md) | 229 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3094/pairing-08cb81327fcb2c5e8bcba94e615183d67828c526.md) | 235 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3094/kitchen_sink-08cb81327fcb2c5e8bcba94e615183d67828c526.md) | 2,032 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/08cb81327fcb2c5e8bcba94e615183d67828c526

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30847184206)
