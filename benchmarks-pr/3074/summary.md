| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/fibonacci-098de5bd9bbba775001c11471156bec162f15c43.md) | 500 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/keccak-098de5bd9bbba775001c11471156bec162f15c43.md) | 7,386 |  14,365,133 |  1,525 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/sha2_bench-098de5bd9bbba775001c11471156bec162f15c43.md) | 4,181 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/regex-098de5bd9bbba775001c11471156bec162f15c43.md) | 687 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/ecrecover-098de5bd9bbba775001c11471156bec162f15c43.md) | 231 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/pairing-098de5bd9bbba775001c11471156bec162f15c43.md) | 237 |  592,827 |  181 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/kitchen_sink-098de5bd9bbba775001c11471156bec162f15c43.md) | 2,052 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/098de5bd9bbba775001c11471156bec162f15c43

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30303503226)
