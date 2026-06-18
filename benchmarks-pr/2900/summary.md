| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2900/fibonacci-fb8eea9f49e6f762162ff235bf11aec9feb62ff2.md) | 3,113 |  12,000,265 |  689 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2900/keccak-fb8eea9f49e6f762162ff235bf11aec9feb62ff2.md) | 16,414 |  18,655,329 |  3,056 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2900/sha2_bench-fb8eea9f49e6f762162ff235bf11aec9feb62ff2.md) | 9,145 |  14,793,960 |  1,112 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2900/regex-fb8eea9f49e6f762162ff235bf11aec9feb62ff2.md) | 1,190 |  4,137,067 |  368 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2900/ecrecover-fb8eea9f49e6f762162ff235bf11aec9feb62ff2.md) | 601 |  123,583 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2900/pairing-fb8eea9f49e6f762162ff235bf11aec9feb62ff2.md) | 935 |  1,745,757 |  308 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2900/kitchen_sink-fb8eea9f49e6f762162ff235bf11aec9feb62ff2.md) | 4,093 |  2,579,903 |  881 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fb8eea9f49e6f762162ff235bf11aec9feb62ff2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27727846932)
