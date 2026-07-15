| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/fibonacci-241d0b93a0438644e0ef9bb70405c1f5ab06398d.md) | 408 |  4,000,051 |  225 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/keccak-241d0b93a0438644e0ef9bb70405c1f5ab06398d.md) | 8,620 |  14,365,133 |  1,543 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/sha2_bench-241d0b93a0438644e0ef9bb70405c1f5ab06398d.md) | 4,127 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/regex-241d0b93a0438644e0ef9bb70405c1f5ab06398d.md) | 492 |  4,090,656 |  187 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/ecrecover-241d0b93a0438644e0ef9bb70405c1f5ab06398d.md) | 225 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/pairing-241d0b93a0438644e0ef9bb70405c1f5ab06398d.md) | 266 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/kitchen_sink-241d0b93a0438644e0ef9bb70405c1f5ab06398d.md) | 1,906 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/241d0b93a0438644e0ef9bb70405c1f5ab06398d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29424400186)
