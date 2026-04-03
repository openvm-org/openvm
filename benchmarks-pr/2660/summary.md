| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2660/fibonacci-5b77043ce8be5bacdd803aa8e8f8171cbf28f9b5.md) | 3,857 |  12,000,265 |  961 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2660/keccak-5b77043ce8be5bacdd803aa8e8f8171cbf28f9b5.md) | 15,769 |  1,235,218 |  2,216 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2660/regex-5b77043ce8be5bacdd803aa8e8f8171cbf28f9b5.md) | 1,452 |  4,136,694 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2660/ecrecover-5b77043ce8be5bacdd803aa8e8f8171cbf28f9b5.md) | 636 |  122,348 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2660/pairing-5b77043ce8be5bacdd803aa8e8f8171cbf28f9b5.md) | 921 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2660/kitchen_sink-5b77043ce8be5bacdd803aa8e8f8171cbf28f9b5.md) | 2,382 |  154,763 |  415 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5b77043ce8be5bacdd803aa8e8f8171cbf28f9b5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23956712280)
