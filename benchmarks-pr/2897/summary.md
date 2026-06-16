| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2897/fibonacci-4d37d8858fe9b2996330cb8db387dbe3cacb95b0.md) | 1,635 |  4,000,051 |  519 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2897/keccak-4d37d8858fe9b2996330cb8db387dbe3cacb95b0.md) | 16,267 |  14,365,133 |  3,029 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2897/sha2_bench-4d37d8858fe9b2996330cb8db387dbe3cacb95b0.md) | 10,400 |  11,167,961 |  1,945 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2897/regex-4d37d8858fe9b2996330cb8db387dbe3cacb95b0.md) | 1,543 |  4,090,656 |  427 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2897/ecrecover-4d37d8858fe9b2996330cb8db387dbe3cacb95b0.md) | 479 |  112,210 |  311 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2897/pairing-4d37d8858fe9b2996330cb8db387dbe3cacb95b0.md) | 620 |  592,827 |  299 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2897/kitchen_sink-4d37d8858fe9b2996330cb8db387dbe3cacb95b0.md) | 3,944 |  1,979,971 |  864 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4d37d8858fe9b2996330cb8db387dbe3cacb95b0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27639498662)
