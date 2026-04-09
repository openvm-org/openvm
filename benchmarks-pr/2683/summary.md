| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/fibonacci-6e0eb764f9416ddf9646636be6b51c77729d37b1.md) | 3,832 |  12,000,265 |  952 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/keccak-6e0eb764f9416ddf9646636be6b51c77729d37b1.md) | 18,635 |  18,655,329 |  3,357 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/regex-6e0eb764f9416ddf9646636be6b51c77729d37b1.md) | 1,418 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/ecrecover-6e0eb764f9416ddf9646636be6b51c77729d37b1.md) | 644 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/pairing-6e0eb764f9416ddf9646636be6b51c77729d37b1.md) | 915 |  1,745,757 |  290 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/kitchen_sink-6e0eb764f9416ddf9646636be6b51c77729d37b1.md) | 2,157 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6e0eb764f9416ddf9646636be6b51c77729d37b1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24196509035)
