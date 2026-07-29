| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/fibonacci-8438fd0f4eaf48ddc054bb60350fc67a116bfb50.md) |<span style='color: green'>(-10 [-0.6%])</span> 1,564 |  12,000,265 | <span style='color: green'>(-5 [-1.4%])</span> 358 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/keccak-8438fd0f4eaf48ddc054bb60350fc67a116bfb50.md) |<span style='color: red'>(+47 [+0.5%])</span> 9,321 |  18,655,329 | <span style='color: red'>(+10 [+0.7%])</span> 1,524 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/sha2_bench-8438fd0f4eaf48ddc054bb60350fc67a116bfb50.md) |<span style='color: red'>(+5 [+0.1%])</span> 4,952 |  14,793,960 | <span style='color: red'>(+5 [+0.9%])</span> 581 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/regex-8438fd0f4eaf48ddc054bb60350fc67a116bfb50.md) |<span style='color: red'>(+3 [+0.5%])</span> 664 |  4,137,067 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/ecrecover-8438fd0f4eaf48ddc054bb60350fc67a116bfb50.md) |<span style='color: red'>(+7 [+1.6%])</span> 438 |  123,583 | <span style='color: red'>(+3 [+1.6%])</span> 185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/pairing-8438fd0f4eaf48ddc054bb60350fc67a116bfb50.md) |<span style='color: red'>(+33 [+5.8%])</span> 602 |  1,745,757 |  190 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/kitchen_sink-8438fd0f4eaf48ddc054bb60350fc67a116bfb50.md) |<span style='color: green'>(-25 [-1.1%])</span> 2,178 |  2,579,903 | <span style='color: green'>(-1 [-0.2%])</span> 474 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8438fd0f4eaf48ddc054bb60350fc67a116bfb50

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30492086388)
