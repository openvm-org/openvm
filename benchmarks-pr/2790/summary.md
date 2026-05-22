| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci-b7b23b65d42b6028d7f1987da95f5d300be9dc36.md) | 3,836 |  12,000,265 |  933 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/keccak-b7b23b65d42b6028d7f1987da95f5d300be9dc36.md) | 19,243 |  18,655,329 |  3,363 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/sha2_bench-b7b23b65d42b6028d7f1987da95f5d300be9dc36.md) | 10,036 |  14,793,960 |  1,437 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex-b7b23b65d42b6028d7f1987da95f5d300be9dc36.md) | 1,394 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover-b7b23b65d42b6028d7f1987da95f5d300be9dc36.md) | 602 |  123,583 |  253 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing-b7b23b65d42b6028d7f1987da95f5d300be9dc36.md) | 886 |  1,745,757 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink-b7b23b65d42b6028d7f1987da95f5d300be9dc36.md) | 1,909 |  2,579,903 |  409 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci_e2e-b7b23b65d42b6028d7f1987da95f5d300be9dc36.md) | 1,777 |  12,000,265 |  411 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex_e2e-b7b23b65d42b6028d7f1987da95f5d300be9dc36.md) | 816 |  4,137,067 |  167 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover_e2e-b7b23b65d42b6028d7f1987da95f5d300be9dc36.md) | 506 |  123,583 |  131 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing_e2e-b7b23b65d42b6028d7f1987da95f5d300be9dc36.md) | 626 |  1,745,757 |  131 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink_e2e-b7b23b65d42b6028d7f1987da95f5d300be9dc36.md) | 2,033 |  2,579,903 |  398 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b7b23b65d42b6028d7f1987da95f5d300be9dc36

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26313639475)
