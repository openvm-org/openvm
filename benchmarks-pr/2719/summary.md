| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci-1f998065deae7e89b008df7fa4287b0acbd7bf6b.md) | 1,554 |  4,000,051 |  435 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/keccak-1f998065deae7e89b008df7fa4287b0acbd7bf6b.md) | 13,727 |  14,365,133 |  2,393 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/sha2_bench-1f998065deae7e89b008df7fa4287b0acbd7bf6b.md) | 8,985 |  11,167,961 |  1,422 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex-1f998065deae7e89b008df7fa4287b0acbd7bf6b.md) | 1,557 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover-1f998065deae7e89b008df7fa4287b0acbd7bf6b.md) | 485 |  112,210 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing-1f998065deae7e89b008df7fa4287b0acbd7bf6b.md) | 603 |  592,827 |  258 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink-1f998065deae7e89b008df7fa4287b0acbd7bf6b.md) | 3,761 |  1,979,971 |  940 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci_e2e-1f998065deae7e89b008df7fa4287b0acbd7bf6b.md) | 812 |  4,000,051 |  197 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex_e2e-1f998065deae7e89b008df7fa4287b0acbd7bf6b.md) | 902 |  4,090,656 |  172 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover_e2e-1f998065deae7e89b008df7fa4287b0acbd7bf6b.md) | 327 |  112,210 |  134 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing_e2e-1f998065deae7e89b008df7fa4287b0acbd7bf6b.md) | 395 |  592,827 |  127 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink_e2e-1f998065deae7e89b008df7fa4287b0acbd7bf6b.md) | 2,051 |  1,979,971 |  393 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1f998065deae7e89b008df7fa4287b0acbd7bf6b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26959801658)
