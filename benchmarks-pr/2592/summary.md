| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-d4bce8212b47e70910e77c9ade44944aa558d0d1.md) | 3,869 |  12,000,265 |  954 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-d4bce8212b47e70910e77c9ade44944aa558d0d1.md) | 18,578 |  18,655,329 |  3,317 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-d4bce8212b47e70910e77c9ade44944aa558d0d1.md) | 1,431 |  4,137,067 |  380 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-d4bce8212b47e70910e77c9ade44944aa558d0d1.md) | 647 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-d4bce8212b47e70910e77c9ade44944aa558d0d1.md) | 908 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-d4bce8212b47e70910e77c9ade44944aa558d0d1.md) | 2,295 |  2,579,903 |  446 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d4bce8212b47e70910e77c9ade44944aa558d0d1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23828258983)
