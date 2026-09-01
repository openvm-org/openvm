| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3144/fibonacci-ac708a3711b7f3dc2e4f8c82fe2468705ebf666c.md) |<span style='color: green'>(-6 [-0.4%])</span> 1,668 |  12,000,265 | <span style='color: green'>(-1 [-0.3%])</span> 368 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3144/keccak-ac708a3711b7f3dc2e4f8c82fe2468705ebf666c.md) |<span style='color: green'>(-92 [-1.0%])</span> 9,503 |  18,655,329 | <span style='color: red'>(+15 [+1.0%])</span> 1,549 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3144/sha2_bench-ac708a3711b7f3dc2e4f8c82fe2468705ebf666c.md) |<span style='color: red'>(+65 [+1.2%])</span> 5,321 |  14,793,960 | <span style='color: red'>(+1 [+0.2%])</span> 591 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3144/regex-ac708a3711b7f3dc2e4f8c82fe2468705ebf666c.md) |<span style='color: red'>(+8 [+1.2%])</span> 702 |  4,137,067 | <span style='color: green'>(-2 [-0.9%])</span> 216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3144/ecrecover-ac708a3711b7f3dc2e4f8c82fe2468705ebf666c.md) |<span style='color: red'>(+4 [+0.9%])</span> 444 |  123,583 | <span style='color: green'>(-4 [-2.1%])</span> 191 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3144/pairing-ac708a3711b7f3dc2e4f8c82fe2468705ebf666c.md) |<span style='color: red'>(+2 [+0.3%])</span> 589 |  1,745,757 | <span style='color: red'>(+2 [+1.0%])</span> 196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3144/kitchen_sink-ac708a3711b7f3dc2e4f8c82fe2468705ebf666c.md) |<span style='color: green'>(-4 [-0.2%])</span> 2,297 |  2,579,903 |  500 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ac708a3711b7f3dc2e4f8c82fe2468705ebf666c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33545594862)
