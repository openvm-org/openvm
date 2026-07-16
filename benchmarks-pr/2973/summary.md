| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/fibonacci-e0a84b56857d6a7abf5c1fa2c320ba31ac977c89.md) | 413 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/keccak-e0a84b56857d6a7abf5c1fa2c320ba31ac977c89.md) | 8,503 |  14,365,133 |  1,537 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/sha2_bench-e0a84b56857d6a7abf5c1fa2c320ba31ac977c89.md) | 4,075 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/regex-e0a84b56857d6a7abf5c1fa2c320ba31ac977c89.md) | 502 |  4,090,656 |  197 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/ecrecover-e0a84b56857d6a7abf5c1fa2c320ba31ac977c89.md) | 223 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/pairing-e0a84b56857d6a7abf5c1fa2c320ba31ac977c89.md) | 268 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/kitchen_sink-e0a84b56857d6a7abf5c1fa2c320ba31ac977c89.md) | 2,014 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e0a84b56857d6a7abf5c1fa2c320ba31ac977c89

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29496590596)
