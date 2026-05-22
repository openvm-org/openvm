| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2810/fibonacci-44d1ea7a1f77d2b7ba8da1448748ea018bbe891d.md) | 3,700 |  12,000,265 |  903 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2810/keccak-44d1ea7a1f77d2b7ba8da1448748ea018bbe891d.md) | 18,596 |  18,655,329 |  3,287 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2810/sha2_bench-44d1ea7a1f77d2b7ba8da1448748ea018bbe891d.md) | 10,085 |  14,793,960 |  1,442 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2810/regex-44d1ea7a1f77d2b7ba8da1448748ea018bbe891d.md) | 1,387 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2810/ecrecover-44d1ea7a1f77d2b7ba8da1448748ea018bbe891d.md) | 597 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2810/pairing-44d1ea7a1f77d2b7ba8da1448748ea018bbe891d.md) | 886 |  1,745,757 |  264 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2810/kitchen_sink-44d1ea7a1f77d2b7ba8da1448748ea018bbe891d.md) | 1,913 |  2,579,903 |  411 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/44d1ea7a1f77d2b7ba8da1448748ea018bbe891d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26312164596)
