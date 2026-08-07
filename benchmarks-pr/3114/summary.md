| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/fibonacci-a31cf6ab46966aeddbd8c66c481a4bc5c5dba8d6.md) | 478 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/keccak-a31cf6ab46966aeddbd8c66c481a4bc5c5dba8d6.md) | 7,355 |  14,365,133 |  1,518 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/sha2_bench-a31cf6ab46966aeddbd8c66c481a4bc5c5dba8d6.md) | 4,220 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/regex-a31cf6ab46966aeddbd8c66c481a4bc5c5dba8d6.md) | 668 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/ecrecover-a31cf6ab46966aeddbd8c66c481a4bc5c5dba8d6.md) | 222 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/pairing-a31cf6ab46966aeddbd8c66c481a4bc5c5dba8d6.md) | 231 |  592,827 |  181 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/kitchen_sink-a31cf6ab46966aeddbd8c66c481a4bc5c5dba8d6.md) | 2,039 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a31cf6ab46966aeddbd8c66c481a4bc5c5dba8d6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31209089574)
