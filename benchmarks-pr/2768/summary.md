| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/fibonacci-1d45a7694e286e8dbfec95826233d3870974cb6e.md) | 4,280 |  12,000,265 |  1,317 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/keccak-1d45a7694e286e8dbfec95826233d3870974cb6e.md) | 21,578 |  18,655,329 |  3,988 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/sha2_bench-1d45a7694e286e8dbfec95826233d3870974cb6e.md) | 10,986 |  14,793,960 |  1,725 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/regex-1d45a7694e286e8dbfec95826233d3870974cb6e.md) | 1,613 |  4,137,067 |  490 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/ecrecover-1d45a7694e286e8dbfec95826233d3870974cb6e.md) | 673 |  123,583 |  356 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/pairing-1d45a7694e286e8dbfec95826233d3870974cb6e.md) | 1,006 |  1,745,757 |  362 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/kitchen_sink-1d45a7694e286e8dbfec95826233d3870974cb6e.md) | 2,322 |  2,579,903 |  654 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/fibonacci_e2e-1d45a7694e286e8dbfec95826233d3870974cb6e.md) | 2,019 |  12,000,265 |  604 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/regex_e2e-1d45a7694e286e8dbfec95826233d3870974cb6e.md) | 914 |  4,137,067 |  240 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/ecrecover_e2e-1d45a7694e286e8dbfec95826233d3870974cb6e.md) | 562 |  123,583 |  188 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/pairing_e2e-1d45a7694e286e8dbfec95826233d3870974cb6e.md) | 694 |  1,745,757 |  188 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/kitchen_sink_e2e-1d45a7694e286e8dbfec95826233d3870974cb6e.md) | 2,433 |  2,579,903 |  645 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1d45a7694e286e8dbfec95826233d3870974cb6e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25384533313)
