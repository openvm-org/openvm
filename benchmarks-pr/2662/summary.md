| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/fibonacci-bd39b6f3cdea7497e09f0b9268e7d382639c3565.md) | 3,808 |  12,000,265 |  944 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/keccak-bd39b6f3cdea7497e09f0b9268e7d382639c3565.md) | 18,447 |  18,655,329 |  3,302 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/regex-bd39b6f3cdea7497e09f0b9268e7d382639c3565.md) | 1,407 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/ecrecover-bd39b6f3cdea7497e09f0b9268e7d382639c3565.md) | 649 |  123,583 |  276 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/pairing-bd39b6f3cdea7497e09f0b9268e7d382639c3565.md) | 908 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/kitchen_sink-bd39b6f3cdea7497e09f0b9268e7d382639c3565.md) | 2,147 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bd39b6f3cdea7497e09f0b9268e7d382639c3565

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24212848298)
