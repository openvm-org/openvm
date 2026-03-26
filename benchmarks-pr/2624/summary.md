| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2624/fibonacci-9818ca09ed823556382cab10cba5b4cd2afb33c6.md) | 3,785 |  12,000,265 |  931 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2624/keccak-9818ca09ed823556382cab10cba5b4cd2afb33c6.md) | 15,639 |  1,235,218 |  2,170 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2624/regex-9818ca09ed823556382cab10cba5b4cd2afb33c6.md) | 1,414 |  4,136,694 |  370 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2624/ecrecover-9818ca09ed823556382cab10cba5b4cd2afb33c6.md) | 637 |  122,348 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2624/pairing-9818ca09ed823556382cab10cba5b4cd2afb33c6.md) | 924 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2624/kitchen_sink-9818ca09ed823556382cab10cba5b4cd2afb33c6.md) | 2,374 |  154,763 |  406 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9818ca09ed823556382cab10cba5b4cd2afb33c6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23613502711)
