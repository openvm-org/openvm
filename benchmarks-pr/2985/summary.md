| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2985/fibonacci-882377c6da3e40ba8010b79a5fa8f5034c0bc471.md) |<span style='color: green'>(-19 [-0.6%])</span> 3,073 |  12,000,265 |  677 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2985/keccak-882377c6da3e40ba8010b79a5fa8f5034c0bc471.md) |<span style='color: green'>(-50 [-0.3%])</span> 16,375 |  18,655,329 | <span style='color: green'>(-12 [-0.4%])</span> 3,034 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2985/sha2_bench-882377c6da3e40ba8010b79a5fa8f5034c0bc471.md) |<span style='color: green'>(-25 [-0.3%])</span> 9,224 |  14,793,960 | <span style='color: red'>(+2 [+0.2%])</span> 1,130 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2985/regex-882377c6da3e40ba8010b79a5fa8f5034c0bc471.md) |<span style='color: green'>(-3 [-0.3%])</span> 1,179 |  4,137,067 | <span style='color: green'>(-3 [-0.8%])</span> 355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2985/ecrecover-882377c6da3e40ba8010b79a5fa8f5034c0bc471.md) | 602 |  123,583 | <span style='color: red'>(+8 [+2.9%])</span> 286 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2985/pairing-882377c6da3e40ba8010b79a5fa8f5034c0bc471.md) |<span style='color: red'>(+12 [+1.3%])</span> 946 |  1,745,757 | <span style='color: red'>(+7 [+2.3%])</span> 312 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2985/kitchen_sink-882377c6da3e40ba8010b79a5fa8f5034c0bc471.md) |<span style='color: green'>(-26 [-0.6%])</span> 4,104 |  2,579,903 | <span style='color: green'>(-7 [-0.8%])</span> 878 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/882377c6da3e40ba8010b79a5fa8f5034c0bc471

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29026538159)
