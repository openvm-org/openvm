| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3138/fibonacci-69feb50e4c1a52dc2e9707acc25b5ea17edb4ce6.md) |<span style='color: red'>(+9 [+0.5%])</span> 1,673 |  12,000,265 | <span style='color: green'>(-2 [-0.5%])</span> 367 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3138/keccak-69feb50e4c1a52dc2e9707acc25b5ea17edb4ce6.md) |<span style='color: red'>(+25 [+0.3%])</span> 9,670 |  18,655,329 | <span style='color: green'>(-4 [-0.3%])</span> 1,547 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3138/sha2_bench-69feb50e4c1a52dc2e9707acc25b5ea17edb4ce6.md) |<span style='color: red'>(+38 [+0.7%])</span> 5,293 |  14,793,960 |  591 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3138/regex-69feb50e4c1a52dc2e9707acc25b5ea17edb4ce6.md) |<span style='color: red'>(+15 [+2.2%])</span> 708 |  4,137,067 | <span style='color: red'>(+3 [+1.4%])</span> 218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3138/ecrecover-69feb50e4c1a52dc2e9707acc25b5ea17edb4ce6.md) |<span style='color: green'>(-3 [-0.7%])</span> 436 |  123,583 | <span style='color: green'>(-3 [-1.6%])</span> 190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3138/pairing-69feb50e4c1a52dc2e9707acc25b5ea17edb4ce6.md) |<span style='color: green'>(-20 [-3.4%])</span> 567 |  1,745,757 | <span style='color: green'>(-5 [-2.5%])</span> 192 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3138/kitchen_sink-69feb50e4c1a52dc2e9707acc25b5ea17edb4ce6.md) |<span style='color: red'>(+14 [+0.6%])</span> 2,309 |  2,579,903 | <span style='color: green'>(-5 [-1.0%])</span> 494 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/69feb50e4c1a52dc2e9707acc25b5ea17edb4ce6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33186748537)
