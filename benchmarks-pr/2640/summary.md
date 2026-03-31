| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2640/fibonacci-5088a292837aa8baf3fb936eb7e5ee4d34f90d0b.md) | 3,811 |  12,000,265 |  939 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2640/keccak-5088a292837aa8baf3fb936eb7e5ee4d34f90d0b.md) | 18,521 |  18,655,329 |  3,297 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2640/regex-5088a292837aa8baf3fb936eb7e5ee4d34f90d0b.md) | 1,423 |  4,137,067 |  370 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2640/ecrecover-5088a292837aa8baf3fb936eb7e5ee4d34f90d0b.md) | 645 |  123,583 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2640/pairing-5088a292837aa8baf3fb936eb7e5ee4d34f90d0b.md) | 893 |  1,745,757 |  277 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2640/kitchen_sink-5088a292837aa8baf3fb936eb7e5ee4d34f90d0b.md) | 2,284 |  2,579,903 |  441 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5088a292837aa8baf3fb936eb7e5ee4d34f90d0b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23821972259)
