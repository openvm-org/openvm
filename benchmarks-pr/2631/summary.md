| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2631/fibonacci-6d8e3b607fea6f0c603a54c066a9ee6a656bd85d.md) | 3,809 |  12,000,265 |  935 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2631/keccak-6d8e3b607fea6f0c603a54c066a9ee6a656bd85d.md) | 18,366 |  18,655,329 |  3,271 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2631/regex-6d8e3b607fea6f0c603a54c066a9ee6a656bd85d.md) | 1,415 |  4,137,067 |  370 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2631/ecrecover-6d8e3b607fea6f0c603a54c066a9ee6a656bd85d.md) | 639 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2631/pairing-6d8e3b607fea6f0c603a54c066a9ee6a656bd85d.md) | 915 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2631/kitchen_sink-6d8e3b607fea6f0c603a54c066a9ee6a656bd85d.md) | 2,282 |  2,579,903 |  438 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6d8e3b607fea6f0c603a54c066a9ee6a656bd85d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23805834351)
