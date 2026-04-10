| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/fibonacci-7d7c9fd4141997439a49e650978c3c0125faab47.md) | 3,812 |  12,000,265 |  952 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/keccak-7d7c9fd4141997439a49e650978c3c0125faab47.md) | 18,647 |  18,655,329 |  3,326 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/regex-7d7c9fd4141997439a49e650978c3c0125faab47.md) | 1,420 |  4,137,067 |  381 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/ecrecover-7d7c9fd4141997439a49e650978c3c0125faab47.md) | 648 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/pairing-7d7c9fd4141997439a49e650978c3c0125faab47.md) | 900 |  1,745,757 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/kitchen_sink-7d7c9fd4141997439a49e650978c3c0125faab47.md) | 2,140 |  2,579,903 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7d7c9fd4141997439a49e650978c3c0125faab47

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24242996126)
