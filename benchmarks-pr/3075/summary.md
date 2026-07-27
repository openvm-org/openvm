| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3075/fibonacci-682bec080e9294c6736e1060d5ae9a580c602157.md) | 469 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3075/keccak-682bec080e9294c6736e1060d5ae9a580c602157.md) | 7,329 |  14,365,133 |  1,533 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3075/sha2_bench-682bec080e9294c6736e1060d5ae9a580c602157.md) | 4,718 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3075/regex-682bec080e9294c6736e1060d5ae9a580c602157.md) | 669 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3075/ecrecover-682bec080e9294c6736e1060d5ae9a580c602157.md) | 228 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3075/pairing-682bec080e9294c6736e1060d5ae9a580c602157.md) | 316 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3075/kitchen_sink-682bec080e9294c6736e1060d5ae9a580c602157.md) | 2,675 |  1,979,971 |  471 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/682bec080e9294c6736e1060d5ae9a580c602157

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30307506575)
