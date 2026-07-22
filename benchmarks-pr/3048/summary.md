| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/fibonacci-2664530c9df20ba9a42961ac69d3b778926bcde1.md) | 476 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/keccak-2664530c9df20ba9a42961ac69d3b778926bcde1.md) | 7,341 |  14,365,133 |  1,537 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/sha2_bench-2664530c9df20ba9a42961ac69d3b778926bcde1.md) | 4,748 |  11,167,961 |  537 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/regex-2664530c9df20ba9a42961ac69d3b778926bcde1.md) | 679 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/ecrecover-2664530c9df20ba9a42961ac69d3b778926bcde1.md) | 228 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/pairing-2664530c9df20ba9a42961ac69d3b778926bcde1.md) | 324 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/kitchen_sink-2664530c9df20ba9a42961ac69d3b778926bcde1.md) | 2,665 |  1,979,971 |  464 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2664530c9df20ba9a42961ac69d3b778926bcde1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29956101329)
