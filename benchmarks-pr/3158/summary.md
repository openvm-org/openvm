| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3158/fibonacci-40edc88806f4baa163cacae83238a83815746db7.md) |<span style='color: green'>(-51 [-3.0%])</span> 1,656 |  12,000,265 | <span style='color: green'>(-11 [-2.9%])</span> 364 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3158/keccak-40edc88806f4baa163cacae83238a83815746db7.md) |<span style='color: red'>(+121 [+1.3%])</span> 9,657 |  18,655,329 | <span style='color: red'>(+16 [+1.0%])</span> 1,565 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3158/sha2_bench-40edc88806f4baa163cacae83238a83815746db7.md) | 5,245 |  14,793,960 | <span style='color: red'>(+1 [+0.2%])</span> 587 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3158/regex-40edc88806f4baa163cacae83238a83815746db7.md) |<span style='color: green'>(-5 [-0.7%])</span> 694 |  4,137,067 | <span style='color: green'>(-2 [-0.9%])</span> 215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3158/ecrecover-40edc88806f4baa163cacae83238a83815746db7.md) | 427 |  123,583 | <span style='color: red'>(+3 [+1.6%])</span> 190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3158/pairing-40edc88806f4baa163cacae83238a83815746db7.md) |<span style='color: red'>(+38 [+6.7%])</span> 603 |  1,745,757 | <span style='color: red'>(+3 [+1.6%])</span> 196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3158/kitchen_sink-40edc88806f4baa163cacae83238a83815746db7.md) |<span style='color: red'>(+11 [+0.5%])</span> 2,318 |  2,579,903 | <span style='color: red'>(+1 [+0.2%])</span> 499 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/40edc88806f4baa163cacae83238a83815746db7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/35004811404)
