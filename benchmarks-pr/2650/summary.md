| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2650/fibonacci-a10b0a98cfcc10fe6df48aabf04c12d178b81436.md) | 3,807 |  12,000,265 |  944 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2650/keccak-a10b0a98cfcc10fe6df48aabf04c12d178b81436.md) | 15,733 |  1,235,218 |  2,199 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2650/regex-a10b0a98cfcc10fe6df48aabf04c12d178b81436.md) | 1,430 |  4,136,694 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2650/ecrecover-a10b0a98cfcc10fe6df48aabf04c12d178b81436.md) | 635 |  122,348 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2650/pairing-a10b0a98cfcc10fe6df48aabf04c12d178b81436.md) | 917 |  1,745,757 |  279 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2650/kitchen_sink-a10b0a98cfcc10fe6df48aabf04c12d178b81436.md) | 2,383 |  154,763 |  414 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a10b0a98cfcc10fe6df48aabf04c12d178b81436

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23870445286)
