| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3140/fibonacci-18f9ab19778ba4030fa3f9e419e4e48cf0d0ecdd.md) |<span style='color: green'>(-15 [-0.9%])</span> 1,680 |  12,000,265 | <span style='color: red'>(+1 [+0.3%])</span> 373 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3140/keccak-18f9ab19778ba4030fa3f9e419e4e48cf0d0ecdd.md) |<span style='color: red'>(+114 [+1.2%])</span> 9,652 |  18,655,329 | <span style='color: red'>(+7 [+0.5%])</span> 1,552 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3140/sha2_bench-18f9ab19778ba4030fa3f9e419e4e48cf0d0ecdd.md) |<span style='color: red'>(+96 [+1.8%])</span> 5,339 |  14,793,960 | <span style='color: red'>(+13 [+2.2%])</span> 599 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3140/regex-18f9ab19778ba4030fa3f9e419e4e48cf0d0ecdd.md) |<span style='color: green'>(-2 [-0.3%])</span> 707 |  4,137,067 | <span style='color: green'>(-3 [-1.4%])</span> 216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3140/ecrecover-18f9ab19778ba4030fa3f9e419e4e48cf0d0ecdd.md) |<span style='color: green'>(-8 [-1.8%])</span> 434 |  123,583 | <span style='color: green'>(-2 [-1.1%])</span> 188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3140/pairing-18f9ab19778ba4030fa3f9e419e4e48cf0d0ecdd.md) |<span style='color: green'>(-8 [-1.4%])</span> 582 |  1,745,757 | <span style='color: green'>(-1 [-0.5%])</span> 195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3140/kitchen_sink-18f9ab19778ba4030fa3f9e419e4e48cf0d0ecdd.md) |<span style='color: red'>(+14 [+0.6%])</span> 2,304 |  2,579,903 | <span style='color: green'>(-4 [-0.8%])</span> 493 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/18f9ab19778ba4030fa3f9e419e4e48cf0d0ecdd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33211865889)
