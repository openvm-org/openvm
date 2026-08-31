| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3141/fibonacci-1dbc26e364091837e1f1f9f3cc1cb16441fb6914.md) | 1,669 |  12,000,265 | <span style='color: green'>(-5 [-1.4%])</span> 364 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3141/keccak-1dbc26e364091837e1f1f9f3cc1cb16441fb6914.md) |<span style='color: red'>(+101 [+1.1%])</span> 9,634 |  18,655,329 | <span style='color: red'>(+15 [+1.0%])</span> 1,558 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3141/sha2_bench-1dbc26e364091837e1f1f9f3cc1cb16441fb6914.md) |<span style='color: green'>(-25 [-0.5%])</span> 5,323 |  14,793,960 | <span style='color: green'>(-5 [-0.8%])</span> 590 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3141/regex-1dbc26e364091837e1f1f9f3cc1cb16441fb6914.md) |<span style='color: green'>(-4 [-0.6%])</span> 700 |  4,137,067 | <span style='color: green'>(-1 [-0.5%])</span> 218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3141/ecrecover-1dbc26e364091837e1f1f9f3cc1cb16441fb6914.md) |<span style='color: green'>(-13 [-3.0%])</span> 425 |  123,583 | <span style='color: green'>(-9 [-4.6%])</span> 188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3141/pairing-1dbc26e364091837e1f1f9f3cc1cb16441fb6914.md) |<span style='color: red'>(+11 [+1.9%])</span> 581 |  1,745,757 |  194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3141/kitchen_sink-1dbc26e364091837e1f1f9f3cc1cb16441fb6914.md) |<span style='color: green'>(-13 [-0.6%])</span> 2,283 |  2,579,903 | <span style='color: green'>(-2 [-0.4%])</span> 497 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1dbc26e364091837e1f1f9f3cc1cb16441fb6914

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33399765151)
