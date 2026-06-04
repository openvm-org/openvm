| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/fibonacci-b1da846e915182f53b7b5f0e663e8bbb03021568.md) | 3,793 |  12,000,265 |  921 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/keccak-b1da846e915182f53b7b5f0e663e8bbb03021568.md) | 18,406 |  18,655,329 |  3,247 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/sha2_bench-b1da846e915182f53b7b5f0e663e8bbb03021568.md) | 10,219 |  14,793,960 |  1,467 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/regex-b1da846e915182f53b7b5f0e663e8bbb03021568.md) | 1,400 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/ecrecover-b1da846e915182f53b7b5f0e663e8bbb03021568.md) | 594 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/pairing-b1da846e915182f53b7b5f0e663e8bbb03021568.md) | 889 |  1,745,757 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/kitchen_sink-b1da846e915182f53b7b5f0e663e8bbb03021568.md) | 1,911 |  2,579,903 |  408 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b1da846e915182f53b7b5f0e663e8bbb03021568

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26975687773)
