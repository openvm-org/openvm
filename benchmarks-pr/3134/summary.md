| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3134/fibonacci-d4930b1c307f5f59a1ab912d2b2208f979333358.md) |<span style='color: red'>(+15 [+0.9%])</span> 1,699 |  12,000,265 | <span style='color: red'>(+3 [+0.8%])</span> 373 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3134/keccak-d4930b1c307f5f59a1ab912d2b2208f979333358.md) |<span style='color: red'>(+186 [+2.0%])</span> 9,704 |  18,655,329 | <span style='color: green'>(-26 [-1.7%])</span> 1,538 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3134/sha2_bench-d4930b1c307f5f59a1ab912d2b2208f979333358.md) |<span style='color: green'>(-127 [-2.4%])</span> 5,230 |  14,793,960 | <span style='color: green'>(-3 [-0.5%])</span> 583 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3134/regex-d4930b1c307f5f59a1ab912d2b2208f979333358.md) |<span style='color: red'>(+13 [+1.9%])</span> 710 |  4,137,067 | <span style='color: red'>(+1 [+0.5%])</span> 217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3134/ecrecover-d4930b1c307f5f59a1ab912d2b2208f979333358.md) |<span style='color: green'>(-3 [-0.7%])</span> 427 |  123,583 | <span style='color: green'>(-2 [-1.1%])</span> 187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3134/pairing-d4930b1c307f5f59a1ab912d2b2208f979333358.md) |<span style='color: green'>(-3 [-0.5%])</span> 587 |  1,745,757 | <span style='color: red'>(+4 [+2.1%])</span> 197 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3134/kitchen_sink-d4930b1c307f5f59a1ab912d2b2208f979333358.md) |<span style='color: green'>(-5 [-0.2%])</span> 2,295 |  2,579,903 | <span style='color: green'>(-3 [-0.6%])</span> 489 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d4930b1c307f5f59a1ab912d2b2208f979333358

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33082920331)
