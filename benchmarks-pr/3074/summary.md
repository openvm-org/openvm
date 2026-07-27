| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/fibonacci-538c5488130da56c8442d33445efe3c1fe5ea8b8.md) | 500 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/keccak-538c5488130da56c8442d33445efe3c1fe5ea8b8.md) | 7,379 |  14,365,133 |  1,516 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/sha2_bench-538c5488130da56c8442d33445efe3c1fe5ea8b8.md) | 4,176 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/regex-538c5488130da56c8442d33445efe3c1fe5ea8b8.md) | 685 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/ecrecover-538c5488130da56c8442d33445efe3c1fe5ea8b8.md) | 235 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/pairing-538c5488130da56c8442d33445efe3c1fe5ea8b8.md) | 239 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/kitchen_sink-538c5488130da56c8442d33445efe3c1fe5ea8b8.md) | 2,035 |  1,979,971 |  456 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/538c5488130da56c8442d33445efe3c1fe5ea8b8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30312743996)
