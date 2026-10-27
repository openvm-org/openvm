| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3180/fibonacci-dc38a7c02d92881885e72a1705308d3d963a07e9.md) |<span style='color: red'>(+10 [+0.6%])</span> 1,696 |  12,000,265 |  367 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3180/keccak-dc38a7c02d92881885e72a1705308d3d963a07e9.md) | 9,683 |  18,655,329 | <span style='color: green'>(-16 [-1.0%])</span> 1,539 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3180/sha2_bench-dc38a7c02d92881885e72a1705308d3d963a07e9.md) |<span style='color: green'>(-65 [-1.2%])</span> 5,275 |  14,793,960 | <span style='color: red'>(+1 [+0.2%])</span> 593 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3180/regex-dc38a7c02d92881885e72a1705308d3d963a07e9.md) |<span style='color: green'>(-12 [-1.7%])</span> 692 |  4,137,067 | <span style='color: green'>(-1 [-0.5%])</span> 219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3180/ecrecover-dc38a7c02d92881885e72a1705308d3d963a07e9.md) |<span style='color: red'>(+5 [+1.1%])</span> 440 |  123,583 | <span style='color: red'>(+3 [+1.6%])</span> 192 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3180/pairing-dc38a7c02d92881885e72a1705308d3d963a07e9.md) |<span style='color: green'>(-39 [-6.4%])</span> 567 |  1,745,757 | <span style='color: red'>(+2 [+1.0%])</span> 196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3180/kitchen_sink-dc38a7c02d92881885e72a1705308d3d963a07e9.md) |<span style='color: red'>(+26 [+1.1%])</span> 2,314 |  2,579,903 | <span style='color: red'>(+3 [+0.6%])</span> 497 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/dc38a7c02d92881885e72a1705308d3d963a07e9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/40587727411)
