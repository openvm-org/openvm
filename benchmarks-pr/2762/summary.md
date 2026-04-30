| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2762/fibonacci-7a57e74947afb152a1f5cf9022008c84b8a3a1a3.md) | 1,882 |  4,000,051 |  530 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2762/keccak-7a57e74947afb152a1f5cf9022008c84b8a3a1a3.md) | 13,468 |  14,365,133 |  2,219 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2762/sha2_bench-7a57e74947afb152a1f5cf9022008c84b8a3a1a3.md) | 9,523 |  11,167,961 |  1,278 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2762/regex-7a57e74947afb152a1f5cf9022008c84b8a3a1a3.md) | 1,586 |  4,090,656 |  380 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2762/ecrecover-7a57e74947afb152a1f5cf9022008c84b8a3a1a3.md) | 645 |  112,210 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2762/pairing-7a57e74947afb152a1f5cf9022008c84b8a3a1a3.md) | 762 |  592,827 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2762/kitchen_sink-7a57e74947afb152a1f5cf9022008c84b8a3a1a3.md) | 2,081 |  1,979,971 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7a57e74947afb152a1f5cf9022008c84b8a3a1a3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25131331946)
