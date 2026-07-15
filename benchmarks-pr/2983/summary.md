| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/fibonacci-269f869defed98196f89fa34cb0ebff9d8437ed0.md) | 407 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/keccak-269f869defed98196f89fa34cb0ebff9d8437ed0.md) | 8,358 |  14,365,133 |  1,511 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/sha2_bench-269f869defed98196f89fa34cb0ebff9d8437ed0.md) | 3,984 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/regex-269f869defed98196f89fa34cb0ebff9d8437ed0.md) | 569 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/ecrecover-269f869defed98196f89fa34cb0ebff9d8437ed0.md) | 218 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/pairing-269f869defed98196f89fa34cb0ebff9d8437ed0.md) | 275 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/kitchen_sink-269f869defed98196f89fa34cb0ebff9d8437ed0.md) | 1,893 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/269f869defed98196f89fa34cb0ebff9d8437ed0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29416510752)
