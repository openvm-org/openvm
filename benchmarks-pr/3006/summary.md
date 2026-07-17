| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/fibonacci-baf0d5fe1ae3ee42a857cbfc6e670e030e708a13.md) |<span style='color: green'>(-59 [-3.7%])</span> 1,545 |  12,000,265 | <span style='color: red'>(+11 [+3.0%])</span> 375 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/keccak-baf0d5fe1ae3ee42a857cbfc6e670e030e708a13.md) |<span style='color: green'>(-1570 [-16.7%])</span> 7,834 |  18,655,329 | <span style='color: green'>(-4 [-0.3%])</span> 1,542 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/sha2_bench-baf0d5fe1ae3ee42a857cbfc6e670e030e708a13.md) |<span style='color: green'>(-294 [-6.1%])</span> 4,560 |  14,793,960 | <span style='color: red'>(+7 [+1.2%])</span> 581 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/regex-baf0d5fe1ae3ee42a857cbfc6e670e030e708a13.md) |<span style='color: red'>(+130 [+19.9%])</span> 783 |  4,137,067 | <span style='color: red'>(+4 [+1.9%])</span> 217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/ecrecover-baf0d5fe1ae3ee42a857cbfc6e670e030e708a13.md) |<span style='color: green'>(-29 [-6.7%])</span> 406 |  123,583 | <span style='color: red'>(+1 [+0.5%])</span> 186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/pairing-baf0d5fe1ae3ee42a857cbfc6e670e030e708a13.md) |<span style='color: red'>(+1 [+0.2%])</span> 599 |  1,745,757 | <span style='color: red'>(+2 [+1.0%])</span> 193 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/kitchen_sink-baf0d5fe1ae3ee42a857cbfc6e670e030e708a13.md) |<span style='color: red'>(+722 [+32.6%])</span> 2,939 |  2,579,903 | <span style='color: red'>(+1 [+0.2%])</span> 480 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/baf0d5fe1ae3ee42a857cbfc6e670e030e708a13

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29554124795)
