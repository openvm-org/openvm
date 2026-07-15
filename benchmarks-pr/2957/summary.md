| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/fibonacci-204ca42e0c698e496ed4f5599f7bcd8ce450ea0e.md) | 407 |  4,000,051 |  226 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/keccak-204ca42e0c698e496ed4f5599f7bcd8ce450ea0e.md) | 8,346 |  14,365,133 |  1,525 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/sha2_bench-204ca42e0c698e496ed4f5599f7bcd8ce450ea0e.md) | 3,941 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/regex-204ca42e0c698e496ed4f5599f7bcd8ce450ea0e.md) | 569 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/ecrecover-204ca42e0c698e496ed4f5599f7bcd8ce450ea0e.md) | 220 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/pairing-204ca42e0c698e496ed4f5599f7bcd8ce450ea0e.md) | 264 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/kitchen_sink-204ca42e0c698e496ed4f5599f7bcd8ce450ea0e.md) | 1,903 |  1,979,971 |  466 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/204ca42e0c698e496ed4f5599f7bcd8ce450ea0e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29411431542)
