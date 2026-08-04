| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-2bc8fde72ea02796a3b8d2a093ea0d46e353ed20.md) | 476 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-2bc8fde72ea02796a3b8d2a093ea0d46e353ed20.md) | 7,380 |  14,365,133 |  1,528 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-2bc8fde72ea02796a3b8d2a093ea0d46e353ed20.md) | 4,152 |  11,167,961 |  534 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-2bc8fde72ea02796a3b8d2a093ea0d46e353ed20.md) | 659 |  4,090,656 |  222 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-2bc8fde72ea02796a3b8d2a093ea0d46e353ed20.md) | 223 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-2bc8fde72ea02796a3b8d2a093ea0d46e353ed20.md) | 234 |  592,827 |  181 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-2bc8fde72ea02796a3b8d2a093ea0d46e353ed20.md) | 2,039 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2bc8fde72ea02796a3b8d2a093ea0d46e353ed20

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30953729775)
