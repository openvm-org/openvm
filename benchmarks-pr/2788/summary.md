| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2788/fibonacci-89469f4568a6e8f4135ba4c2f86971403e09a20e.md) | 3,790 |  12,000,265 |  927 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2788/keccak-89469f4568a6e8f4135ba4c2f86971403e09a20e.md) | 18,670 |  18,655,329 |  3,287 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2788/sha2_bench-89469f4568a6e8f4135ba4c2f86971403e09a20e.md) | 10,166 |  14,793,960 |  1,469 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2788/regex-89469f4568a6e8f4135ba4c2f86971403e09a20e.md) | 1,396 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2788/ecrecover-89469f4568a6e8f4135ba4c2f86971403e09a20e.md) | 598 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2788/pairing-89469f4568a6e8f4135ba4c2f86971403e09a20e.md) | 889 |  1,745,757 |  266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2788/kitchen_sink-89469f4568a6e8f4135ba4c2f86971403e09a20e.md) | 1,892 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/89469f4568a6e8f4135ba4c2f86971403e09a20e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25954772843)
