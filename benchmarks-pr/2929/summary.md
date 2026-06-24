| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2929/fibonacci-c55ccb3e4eecbd5dfbc84cf8cfa8e5414cddc140.md) | 3,077 |  12,000,265 |  680 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2929/keccak-c55ccb3e4eecbd5dfbc84cf8cfa8e5414cddc140.md) | 16,267 |  18,655,329 |  3,009 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2929/sha2_bench-c55ccb3e4eecbd5dfbc84cf8cfa8e5414cddc140.md) | 9,080 |  14,793,960 |  1,109 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2929/regex-c55ccb3e4eecbd5dfbc84cf8cfa8e5414cddc140.md) | 1,161 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2929/ecrecover-c55ccb3e4eecbd5dfbc84cf8cfa8e5414cddc140.md) | 601 |  123,583 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2929/pairing-c55ccb3e4eecbd5dfbc84cf8cfa8e5414cddc140.md) | 931 |  1,745,757 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2929/kitchen_sink-c55ccb3e4eecbd5dfbc84cf8cfa8e5414cddc140.md) | 4,096 |  2,579,903 |  878 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c55ccb3e4eecbd5dfbc84cf8cfa8e5414cddc140

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28132364564)
