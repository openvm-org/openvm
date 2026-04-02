| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-33d732528149a322b7c71a2dd9bcb4d4ed95275e.md) | 3,795 |  12,000,265 |  930 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-33d732528149a322b7c71a2dd9bcb4d4ed95275e.md) | 18,527 |  18,655,329 |  3,309 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-33d732528149a322b7c71a2dd9bcb4d4ed95275e.md) | 1,413 |  4,137,067 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-33d732528149a322b7c71a2dd9bcb4d4ed95275e.md) | 646 |  123,583 |  265 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-33d732528149a322b7c71a2dd9bcb4d4ed95275e.md) | 906 |  1,745,757 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-33d732528149a322b7c71a2dd9bcb4d4ed95275e.md) | 2,276 |  2,579,903 |  439 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/33d732528149a322b7c71a2dd9bcb4d4ed95275e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23905020824)
