| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/fibonacci-b8f7630753585965d98cde7ff9d7528b703a503e.md) |<span style='color: red'>(+5 [+0.3%])</span> 1,684 |  12,000,265 |  371 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/keccak-b8f7630753585965d98cde7ff9d7528b703a503e.md) |<span style='color: red'>(+87 [+0.9%])</span> 9,607 |  18,655,329 | <span style='color: red'>(+4 [+0.3%])</span> 1,543 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/sha2_bench-b8f7630753585965d98cde7ff9d7528b703a503e.md) |<span style='color: green'>(-65 [-1.2%])</span> 5,255 |  14,793,960 | <span style='color: green'>(-1 [-0.2%])</span> 593 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/regex-b8f7630753585965d98cde7ff9d7528b703a503e.md) |<span style='color: green'>(-6 [-0.9%])</span> 692 |  4,137,067 | <span style='color: red'>(+1 [+0.5%])</span> 216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/ecrecover-b8f7630753585965d98cde7ff9d7528b703a503e.md) |<span style='color: green'>(-5 [-1.1%])</span> 434 |  123,583 | <span style='color: green'>(-2 [-1.1%])</span> 188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/pairing-b8f7630753585965d98cde7ff9d7528b703a503e.md) | 577 |  1,745,757 | <span style='color: red'>(+2 [+1.0%])</span> 196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/kitchen_sink-b8f7630753585965d98cde7ff9d7528b703a503e.md) |<span style='color: green'>(-22 [-0.9%])</span> 2,300 |  2,579,903 | <span style='color: green'>(-3 [-0.6%])</span> 493 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b8f7630753585965d98cde7ff9d7528b703a503e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/36138716853)
