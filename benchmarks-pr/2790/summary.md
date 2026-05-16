| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci-ee00d8a5f7bd0fdd6cf27488e98d5d26657ef0c0.md) | 3,758 |  12,000,265 |  918 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/keccak-ee00d8a5f7bd0fdd6cf27488e98d5d26657ef0c0.md) | 18,630 |  18,655,329 |  3,280 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/sha2_bench-ee00d8a5f7bd0fdd6cf27488e98d5d26657ef0c0.md) | 10,135 |  14,793,960 |  1,448 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex-ee00d8a5f7bd0fdd6cf27488e98d5d26657ef0c0.md) | 1,388 |  4,137,067 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover-ee00d8a5f7bd0fdd6cf27488e98d5d26657ef0c0.md) | 598 |  123,583 |  248 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing-ee00d8a5f7bd0fdd6cf27488e98d5d26657ef0c0.md) | 889 |  1,745,757 |  263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink-ee00d8a5f7bd0fdd6cf27488e98d5d26657ef0c0.md) | 1,900 |  2,579,903 |  417 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci_e2e-ee00d8a5f7bd0fdd6cf27488e98d5d26657ef0c0.md) | 1,774 |  12,000,265 |  406 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex_e2e-ee00d8a5f7bd0fdd6cf27488e98d5d26657ef0c0.md) | 818 |  4,137,067 |  170 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover_e2e-ee00d8a5f7bd0fdd6cf27488e98d5d26657ef0c0.md) | 509 |  123,583 |  130 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing_e2e-ee00d8a5f7bd0fdd6cf27488e98d5d26657ef0c0.md) | 634 |  1,745,757 |  131 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink_e2e-ee00d8a5f7bd0fdd6cf27488e98d5d26657ef0c0.md) | 2,020 |  2,579,903 |  400 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ee00d8a5f7bd0fdd6cf27488e98d5d26657ef0c0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25959886689)
