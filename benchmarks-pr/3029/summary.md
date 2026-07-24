| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/fibonacci-aeb4a11ee440415740bdcafec9271e3192b427fd.md) |<span style='color: green'>(-9 [-0.6%])</span> 1,577 |  12,000,265 | <span style='color: green'>(-3 [-0.8%])</span> 358 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/keccak-aeb4a11ee440415740bdcafec9271e3192b427fd.md) |<span style='color: red'>(+36 [+0.4%])</span> 9,291 |  18,655,329 |  1,514 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/sha2_bench-aeb4a11ee440415740bdcafec9271e3192b427fd.md) |<span style='color: red'>(+89 [+1.8%])</span> 4,964 |  14,793,960 | <span style='color: red'>(+5 [+0.9%])</span> 577 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/regex-aeb4a11ee440415740bdcafec9271e3192b427fd.md) |<span style='color: green'>(-6 [-0.9%])</span> 656 |  4,137,067 | <span style='color: red'>(+5 [+2.4%])</span> 215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/ecrecover-aeb4a11ee440415740bdcafec9271e3192b427fd.md) |<span style='color: red'>(+10 [+2.3%])</span> 437 |  123,583 | <span style='color: red'>(+1 [+0.5%])</span> 186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/pairing-aeb4a11ee440415740bdcafec9271e3192b427fd.md) |<span style='color: green'>(-1 [-0.2%])</span> 569 |  1,745,757 | <span style='color: green'>(-1 [-0.5%])</span> 191 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/kitchen_sink-aeb4a11ee440415740bdcafec9271e3192b427fd.md) |<span style='color: green'>(-19 [-0.9%])</span> 2,195 |  2,579,903 | <span style='color: green'>(-2 [-0.4%])</span> 475 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/aeb4a11ee440415740bdcafec9271e3192b427fd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30123755494)
