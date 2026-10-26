| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3176/fibonacci-6c5a794d6769d2c673a6ff93da1f64b7453441eb.md) | 1,686 |  12,000,265 | <span style='color: red'>(+2 [+0.5%])</span> 369 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3176/keccak-6c5a794d6769d2c673a6ff93da1f64b7453441eb.md) | 9,673 |  18,655,329 |  1,554 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3176/sha2_bench-6c5a794d6769d2c673a6ff93da1f64b7453441eb.md) |<span style='color: green'>(-40 [-0.7%])</span> 5,300 |  14,793,960 | <span style='color: green'>(-2 [-0.3%])</span> 590 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3176/regex-6c5a794d6769d2c673a6ff93da1f64b7453441eb.md) |<span style='color: green'>(-4 [-0.6%])</span> 700 |  4,137,067 | <span style='color: green'>(-5 [-2.3%])</span> 215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3176/ecrecover-6c5a794d6769d2c673a6ff93da1f64b7453441eb.md) |<span style='color: red'>(+2 [+0.5%])</span> 437 |  123,583 | <span style='color: red'>(+1 [+0.5%])</span> 190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3176/pairing-6c5a794d6769d2c673a6ff93da1f64b7453441eb.md) |<span style='color: red'>(+3 [+0.5%])</span> 609 |  1,745,757 | <span style='color: red'>(+2 [+1.0%])</span> 196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3176/kitchen_sink-6c5a794d6769d2c673a6ff93da1f64b7453441eb.md) | 2,288 |  2,579,903 | <span style='color: red'>(+2 [+0.4%])</span> 496 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6c5a794d6769d2c673a6ff93da1f64b7453441eb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/40398916980)
