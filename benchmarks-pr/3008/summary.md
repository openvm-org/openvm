| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3008/fibonacci-71e9ede099adc321d6cc15f40138543bd3ff5f2a.md) | 926 |  4,000,051 |  398 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3008/keccak-71e9ede099adc321d6cc15f40138543bd3ff5f2a.md) | 14,909 |  14,365,133 |  3,033 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3008/sha2_bench-71e9ede099adc321d6cc15f40138543bd3ff5f2a.md) | 8,336 |  11,167,961 |  995 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3008/regex-71e9ede099adc321d6cc15f40138543bd3ff5f2a.md) | 1,108 |  4,090,656 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3008/ecrecover-71e9ede099adc321d6cc15f40138543bd3ff5f2a.md) | 312 |  112,210 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3008/pairing-71e9ede099adc321d6cc15f40138543bd3ff5f2a.md) | 475 |  592,827 |  297 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3008/kitchen_sink-71e9ede099adc321d6cc15f40138543bd3ff5f2a.md) | 4,168 |  1,979,971 |  867 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/71e9ede099adc321d6cc15f40138543bd3ff5f2a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29266521204)
