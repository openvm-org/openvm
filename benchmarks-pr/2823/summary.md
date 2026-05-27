| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2823/fibonacci-95d61bcb36935b6d5dead5c2f5c07b8b0cd2989f.md) | 3,768 |  12,000,265 |  917 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2823/keccak-95d61bcb36935b6d5dead5c2f5c07b8b0cd2989f.md) | 18,763 |  18,655,329 |  3,312 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2823/sha2_bench-95d61bcb36935b6d5dead5c2f5c07b8b0cd2989f.md) | 10,125 |  14,793,960 |  1,450 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2823/regex-95d61bcb36935b6d5dead5c2f5c07b8b0cd2989f.md) | 1,399 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2823/ecrecover-95d61bcb36935b6d5dead5c2f5c07b8b0cd2989f.md) | 602 |  123,583 |  251 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2823/pairing-95d61bcb36935b6d5dead5c2f5c07b8b0cd2989f.md) | 880 |  1,745,757 |  260 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2823/kitchen_sink-95d61bcb36935b6d5dead5c2f5c07b8b0cd2989f.md) | 1,914 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/95d61bcb36935b6d5dead5c2f5c07b8b0cd2989f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26543945096)
