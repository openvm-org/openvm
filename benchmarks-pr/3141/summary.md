| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3141/fibonacci-2598e9faf026fbf18bb21043d7a87fde4de9660d.md) |<span style='color: green'>(-13 [-0.8%])</span> 1,656 |  12,000,265 | <span style='color: red'>(+1 [+0.3%])</span> 370 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3141/keccak-2598e9faf026fbf18bb21043d7a87fde4de9660d.md) |<span style='color: red'>(+127 [+1.3%])</span> 9,660 |  18,655,329 |  1,544 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3141/sha2_bench-2598e9faf026fbf18bb21043d7a87fde4de9660d.md) | 5,347 |  14,793,960 |  595 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3141/regex-2598e9faf026fbf18bb21043d7a87fde4de9660d.md) |<span style='color: green'>(-19 [-2.7%])</span> 685 |  4,137,067 | <span style='color: green'>(-5 [-2.3%])</span> 214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3141/ecrecover-2598e9faf026fbf18bb21043d7a87fde4de9660d.md) |<span style='color: green'>(-9 [-2.1%])</span> 429 |  123,583 | <span style='color: green'>(-5 [-2.5%])</span> 192 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3141/pairing-2598e9faf026fbf18bb21043d7a87fde4de9660d.md) |<span style='color: red'>(+34 [+6.0%])</span> 604 |  1,745,757 | <span style='color: red'>(+2 [+1.0%])</span> 196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3141/kitchen_sink-2598e9faf026fbf18bb21043d7a87fde4de9660d.md) | 2,298 |  2,579,903 | <span style='color: green'>(-8 [-1.6%])</span> 491 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2598e9faf026fbf18bb21043d7a87fde4de9660d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33398335562)
