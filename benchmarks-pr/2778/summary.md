| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-61f9e07e58c0b2ff5ee2c14c6f00968177be7ca3.md) | 1,401 |  4,000,051 |  435 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-61f9e07e58c0b2ff5ee2c14c6f00968177be7ca3.md) | 13,217 |  14,365,133 |  2,193 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-61f9e07e58c0b2ff5ee2c14c6f00968177be7ca3.md) | 9,057 |  11,167,961 |  1,418 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-61f9e07e58c0b2ff5ee2c14c6f00968177be7ca3.md) | 1,351 |  4,090,656 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-61f9e07e58c0b2ff5ee2c14c6f00968177be7ca3.md) | 465 |  112,210 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-61f9e07e58c0b2ff5ee2c14c6f00968177be7ca3.md) | 584 |  592,827 |  259 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-61f9e07e58c0b2ff5ee2c14c6f00968177be7ca3.md) | 2,198 |  1,979,971 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/61f9e07e58c0b2ff5ee2c14c6f00968177be7ca3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25969012493)
