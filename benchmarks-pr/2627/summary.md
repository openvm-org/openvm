| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/fibonacci-2bd7eddcd057b4e4595c9d5ee8d855c2b19a45f7.md) | 3,816 |  12,000,265 |  943 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/keccak-2bd7eddcd057b4e4595c9d5ee8d855c2b19a45f7.md) | 15,696 |  1,235,218 |  2,192 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/regex-2bd7eddcd057b4e4595c9d5ee8d855c2b19a45f7.md) | 1,449 |  4,136,694 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/ecrecover-2bd7eddcd057b4e4595c9d5ee8d855c2b19a45f7.md) | 633 |  122,348 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/pairing-2bd7eddcd057b4e4595c9d5ee8d855c2b19a45f7.md) | 919 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/kitchen_sink-2bd7eddcd057b4e4595c9d5ee8d855c2b19a45f7.md) | 2,379 |  154,763 |  415 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2bd7eddcd057b4e4595c9d5ee8d855c2b19a45f7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23654075504)
