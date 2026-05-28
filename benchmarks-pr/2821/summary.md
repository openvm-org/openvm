| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2821/fibonacci-1c34b648717fcd60304cdcddf3ff74c13d4e03b2.md) | 3,757 |  12,000,265 |  917 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2821/keccak-1c34b648717fcd60304cdcddf3ff74c13d4e03b2.md) | 18,579 |  18,655,329 |  3,275 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2821/sha2_bench-1c34b648717fcd60304cdcddf3ff74c13d4e03b2.md) | 10,122 |  14,793,960 |  1,443 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2821/regex-1c34b648717fcd60304cdcddf3ff74c13d4e03b2.md) | 1,400 |  4,137,067 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2821/ecrecover-1c34b648717fcd60304cdcddf3ff74c13d4e03b2.md) | 599 |  123,583 |  245 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2821/pairing-1c34b648717fcd60304cdcddf3ff74c13d4e03b2.md) | 889 |  1,745,757 |  264 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2821/kitchen_sink-1c34b648717fcd60304cdcddf3ff74c13d4e03b2.md) | 1,895 |  2,579,903 |  414 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1c34b648717fcd60304cdcddf3ff74c13d4e03b2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26601713689)
