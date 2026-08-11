| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3117/fibonacci-4bc585b239f1ad8e24e766e6231c0012cb9fd438.md) |<span style='color: green'>(-10 [-0.6%])</span> 1,559 |  12,000,265 | <span style='color: green'>(-3 [-0.8%])</span> 357 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3117/keccak-4bc585b239f1ad8e24e766e6231c0012cb9fd438.md) |<span style='color: green'>(-139 [-1.5%])</span> 9,208 |  18,655,329 | <span style='color: red'>(+18 [+1.2%])</span> 1,535 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3117/sha2_bench-4bc585b239f1ad8e24e766e6231c0012cb9fd438.md) |<span style='color: red'>(+12 [+0.2%])</span> 4,948 |  14,793,960 | <span style='color: red'>(+2 [+0.3%])</span> 579 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3117/regex-4bc585b239f1ad8e24e766e6231c0012cb9fd438.md) | 667 |  4,137,067 | <span style='color: red'>(+2 [+0.9%])</span> 214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3117/ecrecover-4bc585b239f1ad8e24e766e6231c0012cb9fd438.md) |<span style='color: red'>(+4 [+0.9%])</span> 436 |  123,583 | <span style='color: red'>(+2 [+1.1%])</span> 188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3117/pairing-4bc585b239f1ad8e24e766e6231c0012cb9fd438.md) |<span style='color: red'>(+27 [+4.6%])</span> 614 |  1,745,757 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3117/kitchen_sink-4bc585b239f1ad8e24e766e6231c0012cb9fd438.md) |<span style='color: green'>(-5 [-0.2%])</span> 2,203 |  2,579,903 |  474 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4bc585b239f1ad8e24e766e6231c0012cb9fd438

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31523011355)
