| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/fibonacci-baafa75602bbad6893b123b1c397826b0c58f4a3.md) |<span style='color: red'>(+20 [+1.3%])</span> 1,594 |  12,000,265 | <span style='color: green'>(-1 [-0.3%])</span> 362 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/keccak-baafa75602bbad6893b123b1c397826b0c58f4a3.md) |<span style='color: red'>(+37 [+0.4%])</span> 9,311 |  18,655,329 | <span style='color: red'>(+17 [+1.1%])</span> 1,531 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/sha2_bench-baafa75602bbad6893b123b1c397826b0c58f4a3.md) |<span style='color: red'>(+19 [+0.4%])</span> 4,966 |  14,793,960 | <span style='color: red'>(+3 [+0.5%])</span> 579 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/regex-baafa75602bbad6893b123b1c397826b0c58f4a3.md) |<span style='color: red'>(+6 [+0.9%])</span> 667 |  4,137,067 | <span style='color: red'>(+1 [+0.5%])</span> 213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/ecrecover-baafa75602bbad6893b123b1c397826b0c58f4a3.md) |<span style='color: red'>(+7 [+1.6%])</span> 438 |  123,583 | <span style='color: red'>(+1 [+0.5%])</span> 183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/pairing-baafa75602bbad6893b123b1c397826b0c58f4a3.md) |<span style='color: green'>(-20 [-3.5%])</span> 549 |  1,745,757 | <span style='color: green'>(-1 [-0.5%])</span> 189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/kitchen_sink-baafa75602bbad6893b123b1c397826b0c58f4a3.md) |<span style='color: red'>(+15 [+0.7%])</span> 2,218 |  2,579,903 | <span style='color: red'>(+6 [+1.3%])</span> 481 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/baafa75602bbad6893b123b1c397826b0c58f4a3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30383159554)
