| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3140/fibonacci-5961cde46d0645752bf1edfdb0583342bbb91949.md) |<span style='color: green'>(-41 [-2.4%])</span> 1,654 |  12,000,265 | <span style='color: green'>(-6 [-1.6%])</span> 366 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3140/keccak-5961cde46d0645752bf1edfdb0583342bbb91949.md) |<span style='color: red'>(+193 [+2.0%])</span> 9,731 |  18,655,329 | <span style='color: red'>(+25 [+1.6%])</span> 1,570 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3140/sha2_bench-5961cde46d0645752bf1edfdb0583342bbb91949.md) |<span style='color: red'>(+70 [+1.3%])</span> 5,313 |  14,793,960 | <span style='color: red'>(+6 [+1.0%])</span> 592 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3140/regex-5961cde46d0645752bf1edfdb0583342bbb91949.md) |<span style='color: green'>(-5 [-0.7%])</span> 704 |  4,137,067 | <span style='color: green'>(-2 [-0.9%])</span> 217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3140/ecrecover-5961cde46d0645752bf1edfdb0583342bbb91949.md) |<span style='color: green'>(-6 [-1.4%])</span> 436 |  123,583 |  190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3140/pairing-5961cde46d0645752bf1edfdb0583342bbb91949.md) |<span style='color: green'>(-27 [-4.6%])</span> 563 |  1,745,757 | <span style='color: green'>(-1 [-0.5%])</span> 195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3140/kitchen_sink-5961cde46d0645752bf1edfdb0583342bbb91949.md) |<span style='color: red'>(+25 [+1.1%])</span> 2,315 |  2,579,903 |  497 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5961cde46d0645752bf1edfdb0583342bbb91949

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33206645950)
