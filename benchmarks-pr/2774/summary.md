| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/fibonacci-e2fe6fbbf0c632ab7c5496d0a826258ed4766048.md) | 3,879 |  12,000,265 | <span style='color: green'>(-9756 [-91.0%])</span> 968 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/keccak-e2fe6fbbf0c632ab7c5496d0a826258ed4766048.md) | 18,507 |  18,655,329 |  3,302 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/sha2_bench-e2fe6fbbf0c632ab7c5496d0a826258ed4766048.md) | 9,079 |  14,793,960 |  1,416 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/regex-e2fe6fbbf0c632ab7c5496d0a826258ed4766048.md) | 1,409 |  4,137,067 | <span style='color: green'>(-27327 [-98.6%])</span> 378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/ecrecover-e2fe6fbbf0c632ab7c5496d0a826258ed4766048.md) | 648 |  123,583 | <span style='color: green'>(-10577 [-97.4%])</span> 280 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/pairing-e2fe6fbbf0c632ab7c5496d0a826258ed4766048.md) | 910 |  1,745,757 | <span style='color: green'>(-13860 [-98.0%])</span> 289 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/kitchen_sink-e2fe6fbbf0c632ab7c5496d0a826258ed4766048.md) | 2,077 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e2fe6fbbf0c632ab7c5496d0a826258ed4766048

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25662700448)
