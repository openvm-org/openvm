| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-e8f5d93c05d4bfc5e48fbb6ac668bd6354ee0fc0.md) | 3,851 |  12,000,265 |  960 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-e8f5d93c05d4bfc5e48fbb6ac668bd6354ee0fc0.md) | 18,662 |  18,655,329 |  3,354 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-e8f5d93c05d4bfc5e48fbb6ac668bd6354ee0fc0.md) | 1,441 |  4,137,067 |  380 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-e8f5d93c05d4bfc5e48fbb6ac668bd6354ee0fc0.md) | 642 |  123,583 |  264 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-e8f5d93c05d4bfc5e48fbb6ac668bd6354ee0fc0.md) | 900 |  1,745,757 |  279 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-e8f5d93c05d4bfc5e48fbb6ac668bd6354ee0fc0.md) | 2,154 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e8f5d93c05d4bfc5e48fbb6ac668bd6354ee0fc0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24244665643)
