| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3005/fibonacci-ecf507c354c0c63d11f0f925fce041cd616912f0.md) |<span style='color: red'>(+7 [+0.2%])</span> 3,041 |  12,000,265 |  671 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3005/keccak-ecf507c354c0c63d11f0f925fce041cd616912f0.md) |<span style='color: green'>(-134 [-0.8%])</span> 16,195 |  18,655,329 | <span style='color: green'>(-31 [-1.0%])</span> 2,997 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3005/sha2_bench-ecf507c354c0c63d11f0f925fce041cd616912f0.md) |<span style='color: red'>(+210 [+2.3%])</span> 9,347 |  14,793,960 | <span style='color: red'>(+15 [+1.3%])</span> 1,138 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3005/regex-ecf507c354c0c63d11f0f925fce041cd616912f0.md) |<span style='color: red'>(+4 [+0.3%])</span> 1,171 |  4,137,067 | <span style='color: green'>(-1 [-0.3%])</span> 350 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3005/ecrecover-ecf507c354c0c63d11f0f925fce041cd616912f0.md) |<span style='color: red'>(+2 [+0.3%])</span> 600 |  123,583 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3005/pairing-ecf507c354c0c63d11f0f925fce041cd616912f0.md) |<span style='color: red'>(+5 [+0.5%])</span> 936 |  1,745,757 | <span style='color: green'>(-2 [-0.6%])</span> 306 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3005/kitchen_sink-ecf507c354c0c63d11f0f925fce041cd616912f0.md) |<span style='color: green'>(-13 [-0.3%])</span> 4,112 |  2,579,903 | <span style='color: green'>(-1 [-0.1%])</span> 879 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ecf507c354c0c63d11f0f925fce041cd616912f0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29252236630)
