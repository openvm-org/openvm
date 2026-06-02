| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/fibonacci-4e162424752c99a8cb443390378d9d7944221a97.md) | 1,573 |  4,000,051 |  436 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/keccak-4e162424752c99a8cb443390378d9d7944221a97.md) | 14,257 |  14,365,133 |  2,401 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/sha2_bench-4e162424752c99a8cb443390378d9d7944221a97.md) | 9,391 |  11,167,961 |  1,424 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/regex-4e162424752c99a8cb443390378d9d7944221a97.md) | 1,613 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/ecrecover-4e162424752c99a8cb443390378d9d7944221a97.md) | 496 |  112,210 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/pairing-4e162424752c99a8cb443390378d9d7944221a97.md) | 606 |  592,827 |  251 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/kitchen_sink-4e162424752c99a8cb443390378d9d7944221a97.md) | 1,826 |  1,979,971 |  407 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4e162424752c99a8cb443390378d9d7944221a97

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26850448980)
