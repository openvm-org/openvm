| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/fibonacci-2ffd70c27c70deb594d35a523c889be6c149751e.md) | 416 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/keccak-2ffd70c27c70deb594d35a523c889be6c149751e.md) | 8,542 |  14,365,133 |  1,519 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/sha2_bench-2ffd70c27c70deb594d35a523c889be6c149751e.md) | 4,181 |  11,167,961 |  519 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/regex-2ffd70c27c70deb594d35a523c889be6c149751e.md) | 576 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/ecrecover-2ffd70c27c70deb594d35a523c889be6c149751e.md) | 219 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/pairing-2ffd70c27c70deb594d35a523c889be6c149751e.md) | 284 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/kitchen_sink-2ffd70c27c70deb594d35a523c889be6c149751e.md) | 1,928 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2ffd70c27c70deb594d35a523c889be6c149751e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29684795417)
