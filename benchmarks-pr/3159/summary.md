| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3159/fibonacci-a2b6d42c31d5f69dd8471e8be9d86c7cee054e70.md) |<span style='color: green'>(-12 [-0.7%])</span> 1,676 |  12,000,265 | <span style='color: green'>(-4 [-1.1%])</span> 368 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3159/keccak-a2b6d42c31d5f69dd8471e8be9d86c7cee054e70.md) |<span style='color: green'>(-166 [-1.7%])</span> 9,587 |  18,655,329 | <span style='color: green'>(-12 [-0.8%])</span> 1,544 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3159/sha2_bench-a2b6d42c31d5f69dd8471e8be9d86c7cee054e70.md) |<span style='color: red'>(+60 [+1.1%])</span> 5,306 |  14,793,960 | <span style='color: red'>(+4 [+0.7%])</span> 594 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3159/regex-a2b6d42c31d5f69dd8471e8be9d86c7cee054e70.md) |<span style='color: green'>(-4 [-0.6%])</span> 695 |  4,137,067 | <span style='color: red'>(+4 [+1.9%])</span> 219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3159/ecrecover-a2b6d42c31d5f69dd8471e8be9d86c7cee054e70.md) |<span style='color: green'>(-4 [-0.9%])</span> 440 |  123,583 | <span style='color: green'>(-4 [-2.1%])</span> 189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3159/pairing-a2b6d42c31d5f69dd8471e8be9d86c7cee054e70.md) |<span style='color: green'>(-9 [-1.6%])</span> 571 |  1,745,757 | <span style='color: green'>(-1 [-0.5%])</span> 195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3159/kitchen_sink-a2b6d42c31d5f69dd8471e8be9d86c7cee054e70.md) |<span style='color: red'>(+19 [+0.8%])</span> 2,302 |  2,579,903 | <span style='color: red'>(+4 [+0.8%])</span> 494 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a2b6d42c31d5f69dd8471e8be9d86c7cee054e70

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/35012109727)
