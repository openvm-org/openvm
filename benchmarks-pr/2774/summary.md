| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/fibonacci-b4b10b823e9751fb133f36269d4905ed00acdae5.md) | 3,829 |  12,000,265 |  952 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/keccak-b4b10b823e9751fb133f36269d4905ed00acdae5.md) | 18,514 |  18,655,329 |  3,306 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/sha2_bench-b4b10b823e9751fb133f36269d4905ed00acdae5.md) | 8,986 |  14,793,960 |  1,399 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/regex-b4b10b823e9751fb133f36269d4905ed00acdae5.md) | 1,394 |  4,137,067 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/ecrecover-b4b10b823e9751fb133f36269d4905ed00acdae5.md) | 640 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/pairing-b4b10b823e9751fb133f36269d4905ed00acdae5.md) | 902 |  1,745,757 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/kitchen_sink-b4b10b823e9751fb133f36269d4905ed00acdae5.md) | 2,091 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b4b10b823e9751fb133f36269d4905ed00acdae5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25715901813)
