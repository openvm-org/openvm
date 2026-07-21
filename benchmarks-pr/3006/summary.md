| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/fibonacci-6f3ca48619c791a18af18a81f96d2e50b4d9254a.md) | 471 |  4,000,051 |  243 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/keccak-6f3ca48619c791a18af18a81f96d2e50b4d9254a.md) | 7,257 |  14,365,133 |  1,518 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/sha2_bench-6f3ca48619c791a18af18a81f96d2e50b4d9254a.md) | 4,679 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/regex-6f3ca48619c791a18af18a81f96d2e50b4d9254a.md) | 662 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/ecrecover-6f3ca48619c791a18af18a81f96d2e50b4d9254a.md) | 229 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/pairing-6f3ca48619c791a18af18a81f96d2e50b4d9254a.md) | 323 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/kitchen_sink-6f3ca48619c791a18af18a81f96d2e50b4d9254a.md) | 2,645 |  1,979,971 |  473 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6f3ca48619c791a18af18a81f96d2e50b4d9254a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29835490235)
