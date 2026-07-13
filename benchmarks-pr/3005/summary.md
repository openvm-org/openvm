| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3005/fibonacci-9766550486ffa8dfaa930928f44f53d9cb224367.md) |<span style='color: red'>(+10 [+0.3%])</span> 3,044 |  12,000,265 | <span style='color: red'>(+1 [+0.1%])</span> 672 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3005/keccak-9766550486ffa8dfaa930928f44f53d9cb224367.md) |<span style='color: green'>(-18 [-0.1%])</span> 16,311 |  18,655,329 |  3,027 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3005/sha2_bench-9766550486ffa8dfaa930928f44f53d9cb224367.md) |<span style='color: red'>(+127 [+1.4%])</span> 9,264 |  14,793,960 | <span style='color: red'>(+17 [+1.5%])</span> 1,140 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3005/regex-9766550486ffa8dfaa930928f44f53d9cb224367.md) |<span style='color: green'>(-3 [-0.3%])</span> 1,164 |  4,137,067 | <span style='color: red'>(+4 [+1.1%])</span> 355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3005/ecrecover-9766550486ffa8dfaa930928f44f53d9cb224367.md) |<span style='color: red'>(+7 [+1.2%])</span> 605 |  123,583 | <span style='color: red'>(+2 [+0.7%])</span> 286 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3005/pairing-9766550486ffa8dfaa930928f44f53d9cb224367.md) | 931 |  1,745,757 | <span style='color: green'>(-5 [-1.6%])</span> 303 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3005/kitchen_sink-9766550486ffa8dfaa930928f44f53d9cb224367.md) |<span style='color: green'>(-17 [-0.4%])</span> 4,108 |  2,579,903 | <span style='color: green'>(-3 [-0.3%])</span> 877 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9766550486ffa8dfaa930928f44f53d9cb224367

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29107255615)
