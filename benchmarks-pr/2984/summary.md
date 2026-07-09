| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/fibonacci-90e5824086f8c3de7dcb9e1648431b0a900dd483.md) | 871 |  4,000,051 |  391 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/keccak-90e5824086f8c3de7dcb9e1648431b0a900dd483.md) | 15,409 |  14,365,133 |  3,045 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/sha2_bench-90e5824086f8c3de7dcb9e1648431b0a900dd483.md) | 8,065 |  11,167,961 |  1,008 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/regex-90e5824086f8c3de7dcb9e1648431b0a900dd483.md) | 1,047 |  4,090,656 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/ecrecover-90e5824086f8c3de7dcb9e1648431b0a900dd483.md) | 275 |  112,210 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/pairing-90e5824086f8c3de7dcb9e1648431b0a900dd483.md) | 368 |  592,827 |  291 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/kitchen_sink-90e5824086f8c3de7dcb9e1648431b0a900dd483.md) | 3,829 |  1,979,971 |  870 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/90e5824086f8c3de7dcb9e1648431b0a900dd483

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29022543765)
