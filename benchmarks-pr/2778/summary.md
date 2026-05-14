| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-db5ae422fd2d7aced19be94d9b779590a33ecb31.md) | 1,583 |  4,000,051 |  455 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-db5ae422fd2d7aced19be94d9b779590a33ecb31.md) | 13,992 |  14,365,133 |  2,435 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-db5ae422fd2d7aced19be94d9b779590a33ecb31.md) | 9,265 |  11,167,961 |  1,421 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-db5ae422fd2d7aced19be94d9b779590a33ecb31.md) | 1,499 |  4,090,656 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-db5ae422fd2d7aced19be94d9b779590a33ecb31.md) | 505 |  112,210 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-db5ae422fd2d7aced19be94d9b779590a33ecb31.md) | 615 |  592,827 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-db5ae422fd2d7aced19be94d9b779590a33ecb31.md) | 1,946 |  1,979,971 |  430 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/db5ae422fd2d7aced19be94d9b779590a33ecb31

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25877125387)
