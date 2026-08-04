| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3101/fibonacci-e298ad0738a0cac5ac3286ef9c6d52f9d043f2e9.md) | 479 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3101/keccak-e298ad0738a0cac5ac3286ef9c6d52f9d043f2e9.md) | 7,424 |  14,365,133 |  1,529 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3101/sha2_bench-e298ad0738a0cac5ac3286ef9c6d52f9d043f2e9.md) | 4,092 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3101/regex-e298ad0738a0cac5ac3286ef9c6d52f9d043f2e9.md) | 660 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3101/ecrecover-e298ad0738a0cac5ac3286ef9c6d52f9d043f2e9.md) | 223 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3101/pairing-e298ad0738a0cac5ac3286ef9c6d52f9d043f2e9.md) | 235 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3101/kitchen_sink-e298ad0738a0cac5ac3286ef9c6d52f9d043f2e9.md) | 2,030 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e298ad0738a0cac5ac3286ef9c6d52f9d043f2e9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30937751913)
