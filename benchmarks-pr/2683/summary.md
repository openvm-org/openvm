| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/fibonacci-38ee8dc84019d9155182b5ae69218d93399a1b42.md) | 3,843 |  12,000,265 |  959 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/keccak-38ee8dc84019d9155182b5ae69218d93399a1b42.md) | 18,405 |  18,655,329 |  3,305 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/regex-38ee8dc84019d9155182b5ae69218d93399a1b42.md) | 1,414 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/ecrecover-38ee8dc84019d9155182b5ae69218d93399a1b42.md) | 643 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/pairing-38ee8dc84019d9155182b5ae69218d93399a1b42.md) | 903 |  1,745,757 |  289 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/kitchen_sink-38ee8dc84019d9155182b5ae69218d93399a1b42.md) | 2,155 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/38ee8dc84019d9155182b5ae69218d93399a1b42

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24200236338)
