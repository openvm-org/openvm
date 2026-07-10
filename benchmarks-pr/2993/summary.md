| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-af1823a8330099a9e0d53d0093dccf9d189c2024.md) | 880 |  4,000,051 |  399 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-af1823a8330099a9e0d53d0093dccf9d189c2024.md) | 15,833 |  14,365,133 |  3,060 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-af1823a8330099a9e0d53d0093dccf9d189c2024.md) | 7,948 |  11,167,961 |  1,008 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-af1823a8330099a9e0d53d0093dccf9d189c2024.md) | 1,053 |  4,090,656 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-af1823a8330099a9e0d53d0093dccf9d189c2024.md) | 306 |  112,210 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-af1823a8330099a9e0d53d0093dccf9d189c2024.md) | 440 |  592,827 |  300 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-af1823a8330099a9e0d53d0093dccf9d189c2024.md) | 3,727 |  1,979,971 |  868 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/af1823a8330099a9e0d53d0093dccf9d189c2024

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29124964685)
