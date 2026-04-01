| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-9d9606682efb386cf5a6f54c604fe5edbf960ac2.md) | 3,836 |  12,000,265 |  943 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-9d9606682efb386cf5a6f54c604fe5edbf960ac2.md) | 18,785 |  18,655,329 |  3,367 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-9d9606682efb386cf5a6f54c604fe5edbf960ac2.md) | 1,433 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-9d9606682efb386cf5a6f54c604fe5edbf960ac2.md) | 651 |  123,583 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-9d9606682efb386cf5a6f54c604fe5edbf960ac2.md) | 910 |  1,745,757 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-9d9606682efb386cf5a6f54c604fe5edbf960ac2.md) | 2,270 |  2,579,903 |  441 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9d9606682efb386cf5a6f54c604fe5edbf960ac2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23848531239)
