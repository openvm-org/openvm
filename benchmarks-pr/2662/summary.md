| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/fibonacci-9c06d1f32db200a55278c65b41089316968b937b.md) | 3,850 |  12,000,265 |  955 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/keccak-9c06d1f32db200a55278c65b41089316968b937b.md) | 18,391 |  18,655,329 |  3,297 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/regex-9c06d1f32db200a55278c65b41089316968b937b.md) | 1,433 |  4,137,067 |  372 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/ecrecover-9c06d1f32db200a55278c65b41089316968b937b.md) | 644 |  123,583 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/pairing-9c06d1f32db200a55278c65b41089316968b937b.md) | 918 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/kitchen_sink-9c06d1f32db200a55278c65b41089316968b937b.md) | 2,303 |  2,579,903 |  447 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9c06d1f32db200a55278c65b41089316968b937b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24037951091)
