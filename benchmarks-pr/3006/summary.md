| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/fibonacci-a566090dd22982e9d6c142f08e61c64d58940c71.md) |<span style='color: green'>(-52 [-3.3%])</span> 1,534 |  12,000,265 | <span style='color: red'>(+15 [+4.2%])</span> 376 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/keccak-a566090dd22982e9d6c142f08e61c64d58940c71.md) |<span style='color: green'>(-1354 [-14.6%])</span> 7,901 |  18,655,329 | <span style='color: red'>(+45 [+3.0%])</span> 1,560 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/sha2_bench-a566090dd22982e9d6c142f08e61c64d58940c71.md) |<span style='color: green'>(-375 [-7.7%])</span> 4,500 |  14,793,960 | <span style='color: red'>(+16 [+2.8%])</span> 588 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/regex-a566090dd22982e9d6c142f08e61c64d58940c71.md) |<span style='color: red'>(+125 [+18.9%])</span> 787 |  4,137,067 | <span style='color: red'>(+9 [+4.3%])</span> 219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/ecrecover-a566090dd22982e9d6c142f08e61c64d58940c71.md) |<span style='color: green'>(-6 [-1.4%])</span> 421 |  123,583 | <span style='color: red'>(+2 [+1.1%])</span> 187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/pairing-a566090dd22982e9d6c142f08e61c64d58940c71.md) |<span style='color: red'>(+4 [+0.7%])</span> 574 |  1,745,757 | <span style='color: red'>(+2 [+1.0%])</span> 194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/kitchen_sink-a566090dd22982e9d6c142f08e61c64d58940c71.md) |<span style='color: red'>(+721 [+32.6%])</span> 2,935 |  2,579,903 | <span style='color: green'>(-1 [-0.2%])</span> 476 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a566090dd22982e9d6c142f08e61c64d58940c71

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29612501162)
