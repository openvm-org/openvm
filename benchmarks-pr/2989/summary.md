| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2989/fibonacci-2ff0e30e2c8edb95a3246c89769b76aa672c5ea8.md) |<span style='color: green'>(-95 [-3.1%])</span> 2,997 |  12,000,265 | <span style='color: green'>(-19 [-2.8%])</span> 658 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2989/keccak-2ff0e30e2c8edb95a3246c89769b76aa672c5ea8.md) |<span style='color: red'>(+89 [+0.5%])</span> 16,514 |  18,655,329 | <span style='color: red'>(+38 [+1.2%])</span> 3,084 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2989/sha2_bench-2ff0e30e2c8edb95a3246c89769b76aa672c5ea8.md) |<span style='color: green'>(-98 [-1.1%])</span> 9,151 |  14,793,960 |  1,128 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2989/regex-2ff0e30e2c8edb95a3246c89769b76aa672c5ea8.md) |<span style='color: green'>(-5 [-0.4%])</span> 1,177 |  4,137,067 | <span style='color: green'>(-6 [-1.7%])</span> 352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2989/ecrecover-2ff0e30e2c8edb95a3246c89769b76aa672c5ea8.md) |<span style='color: green'>(-31 [-5.1%])</span> 571 |  123,583 | <span style='color: green'>(-5 [-1.8%])</span> 273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2989/pairing-2ff0e30e2c8edb95a3246c89769b76aa672c5ea8.md) |<span style='color: green'>(-90 [-9.6%])</span> 844 |  1,745,757 | <span style='color: green'>(-2 [-0.7%])</span> 303 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2989/kitchen_sink-2ff0e30e2c8edb95a3246c89769b76aa672c5ea8.md) |<span style='color: red'>(+93 [+2.3%])</span> 4,223 |  2,579,903 | <span style='color: red'>(+5 [+0.6%])</span> 890 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2ff0e30e2c8edb95a3246c89769b76aa672c5ea8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29038521114)
