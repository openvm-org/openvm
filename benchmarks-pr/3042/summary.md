| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/fibonacci-2ef90d9029bd9181e83ab75786d4241c7c8b4b38.md) | 413 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/keccak-2ef90d9029bd9181e83ab75786d4241c7c8b4b38.md) | 8,790 |  14,365,133 |  1,550 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/sha2_bench-2ef90d9029bd9181e83ab75786d4241c7c8b4b38.md) | 4,233 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/regex-2ef90d9029bd9181e83ab75786d4241c7c8b4b38.md) | 578 |  4,090,656 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/ecrecover-2ef90d9029bd9181e83ab75786d4241c7c8b4b38.md) | 217 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/pairing-2ef90d9029bd9181e83ab75786d4241c7c8b4b38.md) | 292 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/kitchen_sink-2ef90d9029bd9181e83ab75786d4241c7c8b4b38.md) | 1,921 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2ef90d9029bd9181e83ab75786d4241c7c8b4b38

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29649689563)
