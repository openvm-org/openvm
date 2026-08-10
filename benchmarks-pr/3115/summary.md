| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3115/fibonacci-73435805366ae6c76b22f17d8a58e8041c56ef46.md) |<span style='color: green'>(-11 [-0.7%])</span> 1,580 |  12,000,265 | <span style='color: red'>(+3 [+0.8%])</span> 362 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3115/keccak-73435805366ae6c76b22f17d8a58e8041c56ef46.md) | 9,405 |  18,655,329 | <span style='color: red'>(+18 [+1.2%])</span> 1,549 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3115/sha2_bench-73435805366ae6c76b22f17d8a58e8041c56ef46.md) |<span style='color: green'>(-77 [-1.5%])</span> 4,922 |  14,793,960 | <span style='color: green'>(-1 [-0.2%])</span> 578 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3115/regex-73435805366ae6c76b22f17d8a58e8041c56ef46.md) |<span style='color: red'>(+4 [+0.6%])</span> 665 |  4,137,067 | <span style='color: red'>(+1 [+0.5%])</span> 215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3115/ecrecover-73435805366ae6c76b22f17d8a58e8041c56ef46.md) |<span style='color: red'>(+10 [+2.4%])</span> 434 |  123,583 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3115/pairing-73435805366ae6c76b22f17d8a58e8041c56ef46.md) |<span style='color: red'>(+11 [+2.0%])</span> 564 |  1,745,757 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3115/kitchen_sink-73435805366ae6c76b22f17d8a58e8041c56ef46.md) |<span style='color: green'>(-21 [-0.9%])</span> 2,212 |  2,579,903 | <span style='color: green'>(-4 [-0.8%])</span> 478 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/73435805366ae6c76b22f17d8a58e8041c56ef46

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31399381229)
