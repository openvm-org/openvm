| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-f1e0ecfeb3a8fe31e1330c7b57ab219478eaaff8.md) | 468 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-f1e0ecfeb3a8fe31e1330c7b57ab219478eaaff8.md) | 7,510 |  14,365,133 |  1,548 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-f1e0ecfeb3a8fe31e1330c7b57ab219478eaaff8.md) | 4,166 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-f1e0ecfeb3a8fe31e1330c7b57ab219478eaaff8.md) | 657 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-f1e0ecfeb3a8fe31e1330c7b57ab219478eaaff8.md) | 196 |  112,210 |  199 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-f1e0ecfeb3a8fe31e1330c7b57ab219478eaaff8.md) | 236 |  592,827 |  197 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-f1e0ecfeb3a8fe31e1330c7b57ab219478eaaff8.md) | 2,024 |  1,979,971 |  525 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f1e0ecfeb3a8fe31e1330c7b57ab219478eaaff8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32063680141)
