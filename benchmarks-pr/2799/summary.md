| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/fibonacci-9fa6b8335decd994e285ea7c6d5a2e6380387c83.md) | 3,721 |  12,000,265 |  909 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/keccak-9fa6b8335decd994e285ea7c6d5a2e6380387c83.md) | 18,637 |  18,655,329 |  3,296 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/sha2_bench-9fa6b8335decd994e285ea7c6d5a2e6380387c83.md) | 10,177 |  14,793,960 |  1,463 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/regex-9fa6b8335decd994e285ea7c6d5a2e6380387c83.md) | 1,394 |  4,137,067 |  350 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/ecrecover-9fa6b8335decd994e285ea7c6d5a2e6380387c83.md) | 599 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/pairing-9fa6b8335decd994e285ea7c6d5a2e6380387c83.md) | 893 |  1,745,757 |  265 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/kitchen_sink-9fa6b8335decd994e285ea7c6d5a2e6380387c83.md) | 1,903 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9fa6b8335decd994e285ea7c6d5a2e6380387c83

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26188428941)
