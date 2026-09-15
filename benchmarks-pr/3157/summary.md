| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3157/fibonacci-07d5bd8b725898e4c0a266cb5591428de00a7080.md) |<span style='color: green'>(-5 [-0.3%])</span> 1,670 |  12,000,265 | <span style='color: green'>(-3 [-0.8%])</span> 369 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3157/keccak-07d5bd8b725898e4c0a266cb5591428de00a7080.md) |<span style='color: red'>(+62 [+0.7%])</span> 9,540 |  18,655,329 | <span style='color: green'>(-4 [-0.3%])</span> 1,536 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3157/sha2_bench-07d5bd8b725898e4c0a266cb5591428de00a7080.md) |<span style='color: green'>(-8 [-0.2%])</span> 5,235 |  14,793,960 | <span style='color: green'>(-3 [-0.5%])</span> 585 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3157/regex-07d5bd8b725898e4c0a266cb5591428de00a7080.md) |<span style='color: green'>(-1 [-0.1%])</span> 691 |  4,137,067 | <span style='color: green'>(-1 [-0.5%])</span> 218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3157/ecrecover-07d5bd8b725898e4c0a266cb5591428de00a7080.md) |<span style='color: red'>(+14 [+3.3%])</span> 442 |  123,583 | <span style='color: red'>(+4 [+2.1%])</span> 191 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3157/pairing-07d5bd8b725898e4c0a266cb5591428de00a7080.md) |<span style='color: green'>(-6 [-1.0%])</span> 583 |  1,745,757 | <span style='color: green'>(-1 [-0.5%])</span> 196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3157/kitchen_sink-07d5bd8b725898e4c0a266cb5591428de00a7080.md) |<span style='color: red'>(+24 [+1.0%])</span> 2,312 |  2,579,903 | <span style='color: red'>(+1 [+0.2%])</span> 498 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/07d5bd8b725898e4c0a266cb5591428de00a7080

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/34982870523)
