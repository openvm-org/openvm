| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/fibonacci-9effefa3d5d7aac59768613412872d0c235b26da.md) | 1,543 |  4,000,051 |  431 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/keccak-9effefa3d5d7aac59768613412872d0c235b26da.md) | 13,938 |  14,365,133 |  2,396 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/sha2_bench-9effefa3d5d7aac59768613412872d0c235b26da.md) | 9,450 |  11,167,961 |  1,431 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/regex-9effefa3d5d7aac59768613412872d0c235b26da.md) | 1,437 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/ecrecover-9effefa3d5d7aac59768613412872d0c235b26da.md) | 472 |  112,210 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/pairing-9effefa3d5d7aac59768613412872d0c235b26da.md) | 600 |  592,827 |  257 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/kitchen_sink-9effefa3d5d7aac59768613412872d0c235b26da.md) | 2,171 |  1,979,971 |  408 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9effefa3d5d7aac59768613412872d0c235b26da

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26193530941)
