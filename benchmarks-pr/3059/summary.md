| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3059/fibonacci-bbf5ba8c0746e6faaac533d4888e0218e3a7888b.md) | 474 |  4,000,051 |  243 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3059/keccak-bbf5ba8c0746e6faaac533d4888e0218e3a7888b.md) | 7,353 |  14,365,133 |  1,540 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3059/sha2_bench-bbf5ba8c0746e6faaac533d4888e0218e3a7888b.md) | 4,705 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3059/regex-bbf5ba8c0746e6faaac533d4888e0218e3a7888b.md) | 666 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3059/ecrecover-bbf5ba8c0746e6faaac533d4888e0218e3a7888b.md) | 230 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3059/pairing-bbf5ba8c0746e6faaac533d4888e0218e3a7888b.md) | 315 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3059/kitchen_sink-bbf5ba8c0746e6faaac533d4888e0218e3a7888b.md) | 2,686 |  1,979,971 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bbf5ba8c0746e6faaac533d4888e0218e3a7888b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29932498227)
