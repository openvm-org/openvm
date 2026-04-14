| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2705/fibonacci-453109ccbf0fac66e44e4f9f3742566ade3bf7a9.md) | 3,840 |  12,000,265 |  951 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2705/keccak-453109ccbf0fac66e44e4f9f3742566ade3bf7a9.md) | 18,623 |  18,655,329 |  3,337 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2705/sha2_bench-453109ccbf0fac66e44e4f9f3742566ade3bf7a9.md) | 10,000 |  14,793,960 |  1,419 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2705/regex-453109ccbf0fac66e44e4f9f3742566ade3bf7a9.md) | 1,429 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2705/ecrecover-453109ccbf0fac66e44e4f9f3742566ade3bf7a9.md) | 644 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2705/pairing-453109ccbf0fac66e44e4f9f3742566ade3bf7a9.md) | 911 |  1,745,757 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2705/kitchen_sink-453109ccbf0fac66e44e4f9f3742566ade3bf7a9.md) | 2,156 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/453109ccbf0fac66e44e4f9f3742566ade3bf7a9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24415827378)
