| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/fibonacci-b9877ff37d94a5b0f101ebba77e367017c00bfc5.md) |<span style='color: green'>(-40 [-2.5%])</span> 1,546 |  12,000,265 | <span style='color: red'>(+14 [+3.9%])</span> 375 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/keccak-b9877ff37d94a5b0f101ebba77e367017c00bfc5.md) |<span style='color: green'>(-1385 [-15.0%])</span> 7,870 |  18,655,329 | <span style='color: red'>(+45 [+3.0%])</span> 1,560 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/sha2_bench-b9877ff37d94a5b0f101ebba77e367017c00bfc5.md) |<span style='color: green'>(-320 [-6.6%])</span> 4,555 |  14,793,960 | <span style='color: red'>(+11 [+1.9%])</span> 583 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/regex-b9877ff37d94a5b0f101ebba77e367017c00bfc5.md) |<span style='color: red'>(+120 [+18.1%])</span> 782 |  4,137,067 | <span style='color: red'>(+3 [+1.4%])</span> 213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/ecrecover-b9877ff37d94a5b0f101ebba77e367017c00bfc5.md) |<span style='color: green'>(-8 [-1.9%])</span> 419 |  123,583 | <span style='color: red'>(+4 [+2.2%])</span> 189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/pairing-b9877ff37d94a5b0f101ebba77e367017c00bfc5.md) |<span style='color: red'>(+10 [+1.8%])</span> 580 |  1,745,757 | <span style='color: red'>(+2 [+1.0%])</span> 194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/kitchen_sink-b9877ff37d94a5b0f101ebba77e367017c00bfc5.md) |<span style='color: red'>(+707 [+31.9%])</span> 2,921 |  2,579,903 | <span style='color: green'>(-1 [-0.2%])</span> 476 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b9877ff37d94a5b0f101ebba77e367017c00bfc5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29612999068)
