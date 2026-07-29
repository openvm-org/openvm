| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/fibonacci-0fd82f7837aa8c7615c7a41f571f607e4360473c.md) | 1,574 |  12,000,265 | <span style='color: green'>(-2 [-0.6%])</span> 361 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/keccak-0fd82f7837aa8c7615c7a41f571f607e4360473c.md) |<span style='color: red'>(+46 [+0.5%])</span> 9,320 |  18,655,329 | <span style='color: red'>(+11 [+0.7%])</span> 1,525 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/sha2_bench-0fd82f7837aa8c7615c7a41f571f607e4360473c.md) |<span style='color: red'>(+92 [+1.9%])</span> 5,039 |  14,793,960 | <span style='color: red'>(+2 [+0.3%])</span> 578 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/regex-0fd82f7837aa8c7615c7a41f571f607e4360473c.md) |<span style='color: green'>(-10 [-1.5%])</span> 651 |  4,137,067 | <span style='color: green'>(-1 [-0.5%])</span> 211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/ecrecover-0fd82f7837aa8c7615c7a41f571f607e4360473c.md) |<span style='color: red'>(+7 [+1.6%])</span> 438 |  123,583 | <span style='color: red'>(+2 [+1.1%])</span> 184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/pairing-0fd82f7837aa8c7615c7a41f571f607e4360473c.md) |<span style='color: red'>(+28 [+4.9%])</span> 597 |  1,745,757 | <span style='color: red'>(+1 [+0.5%])</span> 191 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/kitchen_sink-0fd82f7837aa8c7615c7a41f571f607e4360473c.md) |<span style='color: red'>(+13 [+0.6%])</span> 2,216 |  2,579,903 | <span style='color: red'>(+4 [+0.8%])</span> 479 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0fd82f7837aa8c7615c7a41f571f607e4360473c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30496648866)
