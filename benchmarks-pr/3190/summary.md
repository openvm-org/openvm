| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3190/fibonacci-77c7a5efbff40029e1c788ae7ddcebd1ddbd915f.md) |<span style='color: green'>(-33 [-1.9%])</span> 1,685 |  12,000,265 | <span style='color: green'>(-1 [-0.3%])</span> 374 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3190/keccak-77c7a5efbff40029e1c788ae7ddcebd1ddbd915f.md) |<span style='color: red'>(+120 [+1.2%])</span> 9,768 |  18,655,329 | <span style='color: red'>(+7 [+0.5%])</span> 1,553 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3190/sha2_bench-77c7a5efbff40029e1c788ae7ddcebd1ddbd915f.md) | 5,319 |  14,793,960 | <span style='color: red'>(+4 [+0.7%])</span> 590 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3190/regex-77c7a5efbff40029e1c788ae7ddcebd1ddbd915f.md) |<span style='color: red'>(+3 [+0.4%])</span> 695 |  4,137,067 | <span style='color: green'>(-1 [-0.5%])</span> 216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3190/ecrecover-77c7a5efbff40029e1c788ae7ddcebd1ddbd915f.md) |<span style='color: green'>(-2 [-0.5%])</span> 437 |  123,583 | <span style='color: green'>(-3 [-1.6%])</span> 190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3190/pairing-77c7a5efbff40029e1c788ae7ddcebd1ddbd915f.md) |<span style='color: red'>(+51 [+9.1%])</span> 612 |  1,745,757 | <span style='color: green'>(-5 [-2.5%])</span> 195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3190/kitchen_sink-77c7a5efbff40029e1c788ae7ddcebd1ddbd915f.md) |<span style='color: green'>(-25 [-1.1%])</span> 2,277 |  2,579,903 | <span style='color: red'>(+1 [+0.2%])</span> 494 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/77c7a5efbff40029e1c788ae7ddcebd1ddbd915f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/41697877668)
