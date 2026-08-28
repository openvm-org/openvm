| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3138/fibonacci-5f9d832f52763d5e96c427abb098ee3155e774ee.md) |<span style='color: green'>(-5 [-0.3%])</span> 1,659 |  12,000,265 | <span style='color: green'>(-1 [-0.3%])</span> 368 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3138/keccak-5f9d832f52763d5e96c427abb098ee3155e774ee.md) |<span style='color: red'>(+101 [+1.0%])</span> 9,746 |  18,655,329 | <span style='color: red'>(+11 [+0.7%])</span> 1,562 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3138/sha2_bench-5f9d832f52763d5e96c427abb098ee3155e774ee.md) |<span style='color: red'>(+18 [+0.3%])</span> 5,273 |  14,793,960 | <span style='color: green'>(-5 [-0.8%])</span> 586 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3138/regex-5f9d832f52763d5e96c427abb098ee3155e774ee.md) | 693 |  4,137,067 | <span style='color: red'>(+3 [+1.4%])</span> 218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3138/ecrecover-5f9d832f52763d5e96c427abb098ee3155e774ee.md) |<span style='color: red'>(+5 [+1.1%])</span> 444 |  123,583 | <span style='color: red'>(+1 [+0.5%])</span> 194 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3138/pairing-5f9d832f52763d5e96c427abb098ee3155e774ee.md) |<span style='color: green'>(-20 [-3.4%])</span> 567 |  1,745,757 | <span style='color: red'>(+1 [+0.5%])</span> 198 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3138/kitchen_sink-5f9d832f52763d5e96c427abb098ee3155e774ee.md) |<span style='color: green'>(-10 [-0.4%])</span> 2,285 |  2,579,903 | <span style='color: green'>(-7 [-1.4%])</span> 492 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5f9d832f52763d5e96c427abb098ee3155e774ee

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33188568619)
