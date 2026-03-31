| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2639/fibonacci-a790275c47689b4d275c4cb31c0eec90fb3bdb96.md) | 3,870 |  12,000,265 |  944 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2639/keccak-a790275c47689b4d275c4cb31c0eec90fb3bdb96.md) | 18,896 |  18,655,329 |  3,382 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2639/regex-a790275c47689b4d275c4cb31c0eec90fb3bdb96.md) | 1,426 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2639/ecrecover-a790275c47689b4d275c4cb31c0eec90fb3bdb96.md) | 662 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2639/pairing-a790275c47689b4d275c4cb31c0eec90fb3bdb96.md) | 851 |  1,745,757 |  279 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2639/kitchen_sink-a790275c47689b4d275c4cb31c0eec90fb3bdb96.md) | 2,348 |  2,579,903 |  442 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a790275c47689b4d275c4cb31c0eec90fb3bdb96

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23820245462)
