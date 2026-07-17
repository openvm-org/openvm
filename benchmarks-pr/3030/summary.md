| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/fibonacci-30f90ff05c00d8f61bc214cdbd59261da285e3ae.md) |<span style='color: green'>(-20 [-1.2%])</span> 1,584 |  12,000,265 | <span style='color: green'>(-1 [-0.3%])</span> 363 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/keccak-30f90ff05c00d8f61bc214cdbd59261da285e3ae.md) |<span style='color: green'>(-30 [-0.3%])</span> 9,374 |  18,655,329 | <span style='color: green'>(-32 [-2.1%])</span> 1,514 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/sha2_bench-30f90ff05c00d8f61bc214cdbd59261da285e3ae.md) |<span style='color: red'>(+104 [+2.1%])</span> 4,958 |  14,793,960 | <span style='color: red'>(+4 [+0.7%])</span> 578 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/regex-30f90ff05c00d8f61bc214cdbd59261da285e3ae.md) |<span style='color: red'>(+10 [+1.5%])</span> 663 |  4,137,067 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/ecrecover-30f90ff05c00d8f61bc214cdbd59261da285e3ae.md) | 435 |  123,583 | <span style='color: red'>(+4 [+2.2%])</span> 189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/pairing-30f90ff05c00d8f61bc214cdbd59261da285e3ae.md) |<span style='color: green'>(-23 [-3.8%])</span> 575 |  1,745,757 | <span style='color: green'>(-3 [-1.6%])</span> 188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/kitchen_sink-30f90ff05c00d8f61bc214cdbd59261da285e3ae.md) |<span style='color: green'>(-24 [-1.1%])</span> 2,193 |  2,579,903 | <span style='color: green'>(-5 [-1.0%])</span> 474 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/30f90ff05c00d8f61bc214cdbd59261da285e3ae

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29593977694)
