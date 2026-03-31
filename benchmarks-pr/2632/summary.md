| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2632/fibonacci-e3b6d2232ba56b2f0a5c738557af3bdc78063c43.md) | 3,801 |  12,000,265 |  936 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2632/keccak-e3b6d2232ba56b2f0a5c738557af3bdc78063c43.md) | 18,348 |  18,655,329 |  3,276 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2632/regex-e3b6d2232ba56b2f0a5c738557af3bdc78063c43.md) | 1,421 |  4,137,067 |  372 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2632/ecrecover-e3b6d2232ba56b2f0a5c738557af3bdc78063c43.md) | 641 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2632/pairing-e3b6d2232ba56b2f0a5c738557af3bdc78063c43.md) | 900 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2632/kitchen_sink-e3b6d2232ba56b2f0a5c738557af3bdc78063c43.md) | 2,259 |  2,579,903 |  442 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e3b6d2232ba56b2f0a5c738557af3bdc78063c43

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23806459115)
