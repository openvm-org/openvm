| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/fibonacci-0cc43072f689a68db87840c10b1f11f1284093ea.md) | 498 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/keccak-0cc43072f689a68db87840c10b1f11f1284093ea.md) | 7,467 |  14,365,133 |  1,540 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/sha2_bench-0cc43072f689a68db87840c10b1f11f1284093ea.md) | 4,167 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/regex-0cc43072f689a68db87840c10b1f11f1284093ea.md) | 684 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/ecrecover-0cc43072f689a68db87840c10b1f11f1284093ea.md) | 229 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/pairing-0cc43072f689a68db87840c10b1f11f1284093ea.md) | 237 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/kitchen_sink-0cc43072f689a68db87840c10b1f11f1284093ea.md) | 2,066 |  1,979,971 |  464 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0cc43072f689a68db87840c10b1f11f1284093ea

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30306817120)
