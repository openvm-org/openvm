| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/fibonacci-d1e935e5f7975d31d73cac6f7eed810e0c9cee4b.md) |<span style='color: green'>(-19 [-1.2%])</span> 1,567 |  12,000,265 |  361 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/keccak-d1e935e5f7975d31d73cac6f7eed810e0c9cee4b.md) | 9,262 |  18,655,329 | <span style='color: red'>(+4 [+0.3%])</span> 1,519 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/sha2_bench-d1e935e5f7975d31d73cac6f7eed810e0c9cee4b.md) |<span style='color: red'>(+92 [+1.9%])</span> 4,967 |  14,793,960 | <span style='color: red'>(+7 [+1.2%])</span> 579 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/regex-d1e935e5f7975d31d73cac6f7eed810e0c9cee4b.md) |<span style='color: green'>(-1 [-0.2%])</span> 661 |  4,137,067 | <span style='color: red'>(+2 [+1.0%])</span> 212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/ecrecover-d1e935e5f7975d31d73cac6f7eed810e0c9cee4b.md) |<span style='color: red'>(+11 [+2.6%])</span> 438 |  123,583 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/pairing-d1e935e5f7975d31d73cac6f7eed810e0c9cee4b.md) |<span style='color: red'>(+12 [+2.1%])</span> 582 |  1,745,757 | <span style='color: red'>(+1 [+0.5%])</span> 193 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/kitchen_sink-d1e935e5f7975d31d73cac6f7eed810e0c9cee4b.md) |<span style='color: green'>(-14 [-0.6%])</span> 2,200 |  2,579,903 | <span style='color: red'>(+2 [+0.4%])</span> 479 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d1e935e5f7975d31d73cac6f7eed810e0c9cee4b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30127456527)
