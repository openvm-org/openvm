| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2720/fibonacci-964bc728ccd36191041b31e1a3a542e591cb9a71.md) | 3,822 |  12,000,265 |  954 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2720/keccak-964bc728ccd36191041b31e1a3a542e591cb9a71.md) | 18,509 |  18,655,329 |  3,303 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2720/sha2_bench-964bc728ccd36191041b31e1a3a542e591cb9a71.md) | 8,930 |  14,793,960 |  1,387 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2720/regex-964bc728ccd36191041b31e1a3a542e591cb9a71.md) | 1,414 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2720/ecrecover-964bc728ccd36191041b31e1a3a542e591cb9a71.md) | 643 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2720/pairing-964bc728ccd36191041b31e1a3a542e591cb9a71.md) | 906 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2720/kitchen_sink-964bc728ccd36191041b31e1a3a542e591cb9a71.md) | 2,085 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/964bc728ccd36191041b31e1a3a542e591cb9a71

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24681632967)
