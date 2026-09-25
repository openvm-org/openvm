| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/fibonacci-de18b206e025bb818e31c757484352f6aad3e662.md) |<span style='color: green'>(-13 [-0.8%])</span> 1,666 |  12,000,265 | <span style='color: green'>(-2 [-0.5%])</span> 369 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/keccak-de18b206e025bb818e31c757484352f6aad3e662.md) |<span style='color: red'>(+155 [+1.6%])</span> 9,675 |  18,655,329 |  1,540 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/sha2_bench-de18b206e025bb818e31c757484352f6aad3e662.md) |<span style='color: green'>(-36 [-0.7%])</span> 5,284 |  14,793,960 | <span style='color: green'>(-4 [-0.7%])</span> 590 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/regex-de18b206e025bb818e31c757484352f6aad3e662.md) |<span style='color: green'>(-3 [-0.4%])</span> 695 |  4,137,067 | <span style='color: red'>(+2 [+0.9%])</span> 217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/ecrecover-de18b206e025bb818e31c757484352f6aad3e662.md) |<span style='color: red'>(+1 [+0.2%])</span> 440 |  123,583 | <span style='color: red'>(+1 [+0.5%])</span> 191 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/pairing-de18b206e025bb818e31c757484352f6aad3e662.md) |<span style='color: red'>(+12 [+2.1%])</span> 589 |  1,745,757 | <span style='color: green'>(-1 [-0.5%])</span> 193 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/kitchen_sink-de18b206e025bb818e31c757484352f6aad3e662.md) |<span style='color: green'>(-12 [-0.5%])</span> 2,310 |  2,579,903 | <span style='color: green'>(-3 [-0.6%])</span> 493 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/de18b206e025bb818e31c757484352f6aad3e662

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/36138325396)
