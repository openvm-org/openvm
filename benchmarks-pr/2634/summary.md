| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2634/fibonacci-1f908e129c8327db5a103bd4e57af094307015a4.md) | 3,846 |  12,000,265 |  943 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2634/keccak-1f908e129c8327db5a103bd4e57af094307015a4.md) | 15,926 |  1,235,218 |  2,230 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2634/regex-1f908e129c8327db5a103bd4e57af094307015a4.md) | 1,403 |  4,136,694 |  365 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2634/ecrecover-1f908e129c8327db5a103bd4e57af094307015a4.md) | 636 |  122,348 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2634/pairing-1f908e129c8327db5a103bd4e57af094307015a4.md) | 924 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2634/kitchen_sink-1f908e129c8327db5a103bd4e57af094307015a4.md) | 2,381 |  154,763 |  414 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1f908e129c8327db5a103bd4e57af094307015a4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23814364858)
