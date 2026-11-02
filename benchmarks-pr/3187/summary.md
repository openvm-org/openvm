| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3187/fibonacci-83011f074d79468881df530877e609a24ab605b6.md) |<span style='color: green'>(-8 [-0.5%])</span> 1,682 |  12,000,265 | <span style='color: green'>(-2 [-0.5%])</span> 367 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3187/keccak-83011f074d79468881df530877e609a24ab605b6.md) |<span style='color: red'>(+73 [+0.8%])</span> 9,733 |  18,655,329 | <span style='color: red'>(+2 [+0.1%])</span> 1,545 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3187/sha2_bench-83011f074d79468881df530877e609a24ab605b6.md) | 5,332 |  14,793,960 | <span style='color: red'>(+1 [+0.2%])</span> 593 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3187/regex-83011f074d79468881df530877e609a24ab605b6.md) |<span style='color: red'>(+17 [+2.5%])</span> 708 |  4,137,067 | <span style='color: red'>(+3 [+1.4%])</span> 218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3187/ecrecover-83011f074d79468881df530877e609a24ab605b6.md) |<span style='color: red'>(+4 [+0.9%])</span> 437 |  123,583 | <span style='color: green'>(-1 [-0.5%])</span> 189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3187/pairing-83011f074d79468881df530877e609a24ab605b6.md) |<span style='color: green'>(-24 [-4.1%])</span> 563 |  1,745,757 | <span style='color: green'>(-1 [-0.5%])</span> 192 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3187/kitchen_sink-83011f074d79468881df530877e609a24ab605b6.md) |<span style='color: red'>(+25 [+1.1%])</span> 2,318 |  2,579,903 | <span style='color: red'>(+6 [+1.2%])</span> 497 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/83011f074d79468881df530877e609a24ab605b6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/41525716882)
