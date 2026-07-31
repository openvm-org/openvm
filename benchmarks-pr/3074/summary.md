| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/fibonacci-721ae0b2c1cd64129974492d01ebf5e99dc14612.md) | 479 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/keccak-721ae0b2c1cd64129974492d01ebf5e99dc14612.md) | 7,286 |  14,365,133 |  1,515 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/sha2_bench-721ae0b2c1cd64129974492d01ebf5e99dc14612.md) | 4,135 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/regex-721ae0b2c1cd64129974492d01ebf5e99dc14612.md) | 658 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/ecrecover-721ae0b2c1cd64129974492d01ebf5e99dc14612.md) | 236 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/pairing-721ae0b2c1cd64129974492d01ebf5e99dc14612.md) | 239 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/kitchen_sink-721ae0b2c1cd64129974492d01ebf5e99dc14612.md) | 2,045 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/721ae0b2c1cd64129974492d01ebf5e99dc14612

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30667069051)
