| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3135/fibonacci-2845f6d97bd29f8b35b1f2f3b292f74198e6d661.md) |<span style='color: red'>(+19 [+1.1%])</span> 1,675 |  12,000,265 | <span style='color: red'>(+2 [+0.5%])</span> 369 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3135/keccak-2845f6d97bd29f8b35b1f2f3b292f74198e6d661.md) |<span style='color: green'>(-54 [-0.6%])</span> 9,555 |  18,655,329 | <span style='color: green'>(-15 [-1.0%])</span> 1,523 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3135/sha2_bench-2845f6d97bd29f8b35b1f2f3b292f74198e6d661.md) |<span style='color: red'>(+86 [+1.6%])</span> 5,347 |  14,793,960 | <span style='color: red'>(+4 [+0.7%])</span> 597 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3135/regex-2845f6d97bd29f8b35b1f2f3b292f74198e6d661.md) |<span style='color: green'>(-6 [-0.8%])</span> 708 |  4,137,067 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3135/ecrecover-2845f6d97bd29f8b35b1f2f3b292f74198e6d661.md) |<span style='color: red'>(+7 [+1.6%])</span> 442 |  123,583 | <span style='color: green'>(-1 [-0.5%])</span> 190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3135/pairing-2845f6d97bd29f8b35b1f2f3b292f74198e6d661.md) |<span style='color: red'>(+42 [+7.3%])</span> 616 |  1,745,757 |  195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3135/kitchen_sink-2845f6d97bd29f8b35b1f2f3b292f74198e6d661.md) |<span style='color: green'>(-8 [-0.3%])</span> 2,282 |  2,579,903 | <span style='color: green'>(-9 [-1.8%])</span> 488 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2845f6d97bd29f8b35b1f2f3b292f74198e6d661

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33099847784)
