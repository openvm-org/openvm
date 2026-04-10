| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/fibonacci-2b7ae501e2547ac565ff461b4bb7b08ad4fa1ee3.md) | 3,836 |  12,000,265 |  950 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/keccak-2b7ae501e2547ac565ff461b4bb7b08ad4fa1ee3.md) | 18,609 |  18,655,329 |  3,339 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/regex-2b7ae501e2547ac565ff461b4bb7b08ad4fa1ee3.md) | 1,421 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/ecrecover-2b7ae501e2547ac565ff461b4bb7b08ad4fa1ee3.md) | 648 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/pairing-2b7ae501e2547ac565ff461b4bb7b08ad4fa1ee3.md) | 908 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/kitchen_sink-2b7ae501e2547ac565ff461b4bb7b08ad4fa1ee3.md) | 2,176 |  2,579,903 |  440 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2b7ae501e2547ac565ff461b4bb7b08ad4fa1ee3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24252408781)
