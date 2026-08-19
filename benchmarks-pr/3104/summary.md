| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-417f315a462c32c633bb9ef75d619947a84ef6b6.md) | 438 |  4,000,051 |  227 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-417f315a462c32c633bb9ef75d619947a84ef6b6.md) | 7,231 |  14,365,133 |  1,623 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-417f315a462c32c633bb9ef75d619947a84ef6b6.md) | 4,090 |  11,167,961 |  518 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-417f315a462c32c633bb9ef75d619947a84ef6b6.md) | 713 |  4,090,656 |  211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-417f315a462c32c633bb9ef75d619947a84ef6b6.md) | 207 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-417f315a462c32c633bb9ef75d619947a84ef6b6.md) | 241 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-417f315a462c32c633bb9ef75d619947a84ef6b6.md) | 2,162 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/417f315a462c32c633bb9ef75d619947a84ef6b6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32281463164)
