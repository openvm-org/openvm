| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/fibonacci-60d9d95ef7d974f9e9979446a8c0d6fdea80d0c4.md) | 3,833 |  12,000,265 |  951 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/keccak-60d9d95ef7d974f9e9979446a8c0d6fdea80d0c4.md) | 18,391 |  18,655,329 |  3,288 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/sha2_bench-60d9d95ef7d974f9e9979446a8c0d6fdea80d0c4.md) | 8,903 |  14,793,960 |  1,378 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/regex-60d9d95ef7d974f9e9979446a8c0d6fdea80d0c4.md) | 1,425 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/ecrecover-60d9d95ef7d974f9e9979446a8c0d6fdea80d0c4.md) | 644 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/pairing-60d9d95ef7d974f9e9979446a8c0d6fdea80d0c4.md) | 904 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/kitchen_sink-60d9d95ef7d974f9e9979446a8c0d6fdea80d0c4.md) | 2,089 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/60d9d95ef7d974f9e9979446a8c0d6fdea80d0c4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24343562411)
