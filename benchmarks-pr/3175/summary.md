| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3175/fibonacci-e736bdd33e734f318d8f47056c5c2fbd0c3e8b82.md) |<span style='color: green'>(-30 [-1.8%])</span> 1,649 |  12,000,265 | <span style='color: green'>(-2 [-0.5%])</span> 369 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3175/keccak-e736bdd33e734f318d8f47056c5c2fbd0c3e8b82.md) |<span style='color: red'>(+137 [+1.4%])</span> 9,649 |  18,655,329 | <span style='color: red'>(+8 [+0.5%])</span> 1,547 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3175/sha2_bench-e736bdd33e734f318d8f47056c5c2fbd0c3e8b82.md) |<span style='color: red'>(+50 [+1.0%])</span> 5,266 |  14,793,960 | <span style='color: red'>(+5 [+0.9%])</span> 589 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3175/regex-e736bdd33e734f318d8f47056c5c2fbd0c3e8b82.md) |<span style='color: red'>(+18 [+2.6%])</span> 700 |  4,137,067 | <span style='color: red'>(+7 [+3.3%])</span> 221 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3175/ecrecover-e736bdd33e734f318d8f47056c5c2fbd0c3e8b82.md) |<span style='color: red'>(+9 [+2.1%])</span> 436 |  123,583 | <span style='color: red'>(+6 [+3.2%])</span> 194 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3175/pairing-e736bdd33e734f318d8f47056c5c2fbd0c3e8b82.md) |<span style='color: red'>(+22 [+3.8%])</span> 604 |  1,745,757 | <span style='color: red'>(+2 [+1.0%])</span> 196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3175/kitchen_sink-e736bdd33e734f318d8f47056c5c2fbd0c3e8b82.md) |<span style='color: red'>(+23 [+1.0%])</span> 2,310 |  2,579,903 |  493 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e736bdd33e734f318d8f47056c5c2fbd0c3e8b82

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/40372788248)
