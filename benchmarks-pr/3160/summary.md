| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/fibonacci-54d4c2792a21464540c2c07227c0af618fd475fa.md) |<span style='color: red'>(+8 [+0.5%])</span> 1,687 |  12,000,265 | <span style='color: red'>(+1 [+0.3%])</span> 372 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/keccak-54d4c2792a21464540c2c07227c0af618fd475fa.md) |<span style='color: red'>(+20 [+0.2%])</span> 9,540 |  18,655,329 | <span style='color: green'>(-10 [-0.6%])</span> 1,529 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/sha2_bench-54d4c2792a21464540c2c07227c0af618fd475fa.md) |<span style='color: green'>(-13 [-0.2%])</span> 5,307 |  14,793,960 | <span style='color: red'>(+3 [+0.5%])</span> 597 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/regex-54d4c2792a21464540c2c07227c0af618fd475fa.md) |<span style='color: green'>(-9 [-1.3%])</span> 689 |  4,137,067 | <span style='color: red'>(+4 [+1.9%])</span> 219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/ecrecover-54d4c2792a21464540c2c07227c0af618fd475fa.md) |<span style='color: green'>(-9 [-2.1%])</span> 430 |  123,583 | <span style='color: green'>(-1 [-0.5%])</span> 189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/pairing-54d4c2792a21464540c2c07227c0af618fd475fa.md) |<span style='color: red'>(+9 [+1.6%])</span> 586 |  1,745,757 |  194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/kitchen_sink-54d4c2792a21464540c2c07227c0af618fd475fa.md) |<span style='color: green'>(-8 [-0.3%])</span> 2,314 |  2,579,903 | <span style='color: green'>(-2 [-0.4%])</span> 494 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/54d4c2792a21464540c2c07227c0af618fd475fa

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/36600613828)
