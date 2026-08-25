| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3131/fibonacci-84f5ea1a71565ec52d3114795cb34ccac0f77f1d.md) |<span style='color: green'>(-30 [-1.8%])</span> 1,670 |  12,000,265 |  372 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3131/keccak-84f5ea1a71565ec52d3114795cb34ccac0f77f1d.md) |<span style='color: green'>(-39 [-0.4%])</span> 9,683 |  18,655,329 | <span style='color: red'>(+3 [+0.2%])</span> 1,543 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3131/sha2_bench-84f5ea1a71565ec52d3114795cb34ccac0f77f1d.md) |<span style='color: green'>(-82 [-1.5%])</span> 5,255 |  14,793,960 | <span style='color: green'>(-1 [-0.2%])</span> 593 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3131/regex-84f5ea1a71565ec52d3114795cb34ccac0f77f1d.md) |<span style='color: red'>(+18 [+2.6%])</span> 711 |  4,137,067 | <span style='color: red'>(+3 [+1.4%])</span> 219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3131/ecrecover-84f5ea1a71565ec52d3114795cb34ccac0f77f1d.md) |<span style='color: red'>(+11 [+2.6%])</span> 439 |  123,583 | <span style='color: red'>(+4 [+2.1%])</span> 193 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3131/pairing-84f5ea1a71565ec52d3114795cb34ccac0f77f1d.md) |<span style='color: green'>(-15 [-2.6%])</span> 564 |  1,745,757 | <span style='color: green'>(-1 [-0.5%])</span> 195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3131/kitchen_sink-84f5ea1a71565ec52d3114795cb34ccac0f77f1d.md) |<span style='color: red'>(+10 [+0.4%])</span> 2,293 |  2,579,903 | <span style='color: red'>(+13 [+2.7%])</span> 502 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/84f5ea1a71565ec52d3114795cb34ccac0f77f1d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32877854773)
