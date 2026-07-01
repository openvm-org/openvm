| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/fibonacci-df248975e03879702577c41d6b4d3ea713d9bcfd.md) | 3,041 |  12,000,265 |  673 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/keccak-df248975e03879702577c41d6b4d3ea713d9bcfd.md) | 16,381 |  18,655,329 |  3,029 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/sha2_bench-df248975e03879702577c41d6b4d3ea713d9bcfd.md) | 9,285 |  14,793,960 |  1,135 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/regex-df248975e03879702577c41d6b4d3ea713d9bcfd.md) | 1,175 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/ecrecover-df248975e03879702577c41d6b4d3ea713d9bcfd.md) | 600 |  123,583 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/pairing-df248975e03879702577c41d6b4d3ea713d9bcfd.md) | 936 |  1,745,757 |  305 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/kitchen_sink-df248975e03879702577c41d6b4d3ea713d9bcfd.md) | 4,117 |  2,579,903 |  881 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/df248975e03879702577c41d6b4d3ea713d9bcfd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28544174571)
