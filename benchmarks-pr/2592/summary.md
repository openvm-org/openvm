| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-db12abcddad13294423c81c139642244a6e95e58.md) | 3,806 |  12,000,265 |  949 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-db12abcddad13294423c81c139642244a6e95e58.md) | 18,537 |  18,655,329 |  3,317 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-db12abcddad13294423c81c139642244a6e95e58.md) | 1,413 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-db12abcddad13294423c81c139642244a6e95e58.md) | 651 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-db12abcddad13294423c81c139642244a6e95e58.md) | 913 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-db12abcddad13294423c81c139642244a6e95e58.md) | 2,293 |  2,579,903 |  447 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/db12abcddad13294423c81c139642244a6e95e58

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24141460267)
