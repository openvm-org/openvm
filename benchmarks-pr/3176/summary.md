| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3176/fibonacci-76f0add7efda96e28c59c9b23501766035536687.md) |<span style='color: green'>(-31 [-1.8%])</span> 1,658 |  12,000,265 | <span style='color: green'>(-11 [-2.9%])</span> 366 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3176/keccak-76f0add7efda96e28c59c9b23501766035536687.md) |<span style='color: red'>(+16 [+0.2%])</span> 9,688 |  18,655,329 |  1,547 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3176/sha2_bench-76f0add7efda96e28c59c9b23501766035536687.md) |<span style='color: red'>(+23 [+0.4%])</span> 5,247 |  14,793,960 |  587 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3176/regex-76f0add7efda96e28c59c9b23501766035536687.md) |<span style='color: green'>(-8 [-1.1%])</span> 695 |  4,137,067 | <span style='color: green'>(-1 [-0.5%])</span> 216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3176/ecrecover-76f0add7efda96e28c59c9b23501766035536687.md) |<span style='color: red'>(+5 [+1.1%])</span> 442 |  123,583 |  189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3176/pairing-76f0add7efda96e28c59c9b23501766035536687.md) |<span style='color: green'>(-16 [-2.7%])</span> 574 |  1,745,757 | <span style='color: green'>(-1 [-0.5%])</span> 194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3176/kitchen_sink-76f0add7efda96e28c59c9b23501766035536687.md) |<span style='color: red'>(+24 [+1.1%])</span> 2,306 |  2,579,903 | <span style='color: red'>(+7 [+1.4%])</span> 498 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/76f0add7efda96e28c59c9b23501766035536687

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/40389412122)
