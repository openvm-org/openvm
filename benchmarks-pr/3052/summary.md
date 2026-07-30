| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/fibonacci-0fd82f7837aa8c7615c7a41f571f607e4360473c.md) |<span style='color: green'>(-5 [-0.3%])</span> 1,569 |  12,000,265 | <span style='color: green'>(-6 [-1.7%])</span> 357 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/keccak-0fd82f7837aa8c7615c7a41f571f607e4360473c.md) |<span style='color: red'>(+25 [+0.3%])</span> 9,299 |  18,655,329 | <span style='color: red'>(+2 [+0.1%])</span> 1,516 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/sha2_bench-0fd82f7837aa8c7615c7a41f571f607e4360473c.md) |<span style='color: red'>(+26 [+0.5%])</span> 4,973 |  14,793,960 | <span style='color: red'>(+1 [+0.2%])</span> 577 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/regex-0fd82f7837aa8c7615c7a41f571f607e4360473c.md) | 661 |  4,137,067 | <span style='color: green'>(-2 [-0.9%])</span> 210 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/ecrecover-0fd82f7837aa8c7615c7a41f571f607e4360473c.md) |<span style='color: red'>(+5 [+1.2%])</span> 436 |  123,583 | <span style='color: red'>(+1 [+0.5%])</span> 183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/pairing-0fd82f7837aa8c7615c7a41f571f607e4360473c.md) |<span style='color: green'>(-11 [-1.9%])</span> 558 |  1,745,757 |  190 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/kitchen_sink-0fd82f7837aa8c7615c7a41f571f607e4360473c.md) |<span style='color: red'>(+5 [+0.2%])</span> 2,208 |  2,579,903 | <span style='color: green'>(-1 [-0.2%])</span> 474 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0fd82f7837aa8c7615c7a41f571f607e4360473c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30496648866)
