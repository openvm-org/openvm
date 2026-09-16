| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3159/fibonacci-987083363941976a294172dce2b98b5a6de52bc5.md) |<span style='color: green'>(-29 [-1.7%])</span> 1,659 |  12,000,265 | <span style='color: green'>(-7 [-1.9%])</span> 365 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3159/keccak-987083363941976a294172dce2b98b5a6de52bc5.md) |<span style='color: green'>(-239 [-2.5%])</span> 9,514 |  18,655,329 | <span style='color: green'>(-31 [-2.0%])</span> 1,525 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3159/sha2_bench-987083363941976a294172dce2b98b5a6de52bc5.md) |<span style='color: red'>(+13 [+0.2%])</span> 5,259 |  14,793,960 | <span style='color: red'>(+1 [+0.2%])</span> 591 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3159/regex-987083363941976a294172dce2b98b5a6de52bc5.md) |<span style='color: green'>(-14 [-2.0%])</span> 685 |  4,137,067 | <span style='color: red'>(+2 [+0.9%])</span> 217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3159/ecrecover-987083363941976a294172dce2b98b5a6de52bc5.md) |<span style='color: green'>(-16 [-3.6%])</span> 428 |  123,583 | <span style='color: green'>(-2 [-1.0%])</span> 191 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3159/pairing-987083363941976a294172dce2b98b5a6de52bc5.md) |<span style='color: green'>(-4 [-0.7%])</span> 576 |  1,745,757 | <span style='color: green'>(-3 [-1.5%])</span> 193 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3159/kitchen_sink-987083363941976a294172dce2b98b5a6de52bc5.md) | 2,283 |  2,579,903 | <span style='color: red'>(+9 [+1.8%])</span> 499 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/987083363941976a294172dce2b98b5a6de52bc5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/35110943433)
