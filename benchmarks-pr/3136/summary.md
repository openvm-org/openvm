| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3136/fibonacci-f188fd788713ad95e5fb18cbcd15c86ab114ff96.md) |<span style='color: red'>(+27 [+1.6%])</span> 1,686 |  12,000,265 | <span style='color: red'>(+2 [+0.5%])</span> 371 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3136/keccak-f188fd788713ad95e5fb18cbcd15c86ab114ff96.md) |<span style='color: green'>(-166 [-1.7%])</span> 9,503 |  18,655,329 | <span style='color: green'>(-15 [-1.0%])</span> 1,530 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3136/sha2_bench-f188fd788713ad95e5fb18cbcd15c86ab114ff96.md) |<span style='color: green'>(-22 [-0.4%])</span> 5,249 |  14,793,960 | <span style='color: red'>(+1 [+0.2%])</span> 588 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3136/regex-f188fd788713ad95e5fb18cbcd15c86ab114ff96.md) |<span style='color: red'>(+4 [+0.6%])</span> 708 |  4,137,067 | <span style='color: red'>(+1 [+0.5%])</span> 218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3136/ecrecover-f188fd788713ad95e5fb18cbcd15c86ab114ff96.md) |<span style='color: red'>(+7 [+1.6%])</span> 439 |  123,583 | <span style='color: green'>(-1 [-0.5%])</span> 193 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3136/pairing-f188fd788713ad95e5fb18cbcd15c86ab114ff96.md) |<span style='color: green'>(-4 [-0.7%])</span> 580 |  1,745,757 | <span style='color: red'>(+4 [+2.1%])</span> 198 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3136/kitchen_sink-f188fd788713ad95e5fb18cbcd15c86ab114ff96.md) |<span style='color: red'>(+19 [+0.8%])</span> 2,319 |  2,579,903 | <span style='color: red'>(+6 [+1.2%])</span> 498 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f188fd788713ad95e5fb18cbcd15c86ab114ff96

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33106339573)
