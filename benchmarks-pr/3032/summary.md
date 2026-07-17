| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/fibonacci-60d1b6a98c727974e7a1906c1e4b38970b018f7f.md) |<span style='color: green'>(-56 [-3.5%])</span> 1,548 |  12,000,265 | <span style='color: green'>(-8 [-2.2%])</span> 356 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/keccak-60d1b6a98c727974e7a1906c1e4b38970b018f7f.md) |<span style='color: green'>(-83 [-0.9%])</span> 9,321 |  18,655,329 | <span style='color: green'>(-19 [-1.2%])</span> 1,527 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/sha2_bench-60d1b6a98c727974e7a1906c1e4b38970b018f7f.md) |<span style='color: red'>(+104 [+2.1%])</span> 4,958 |  14,793,960 | <span style='color: red'>(+11 [+1.9%])</span> 585 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/regex-60d1b6a98c727974e7a1906c1e4b38970b018f7f.md) |<span style='color: green'>(-2 [-0.3%])</span> 651 |  4,137,067 | <span style='color: green'>(-2 [-0.9%])</span> 211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/ecrecover-60d1b6a98c727974e7a1906c1e4b38970b018f7f.md) |<span style='color: red'>(+4 [+0.9%])</span> 439 |  123,583 | <span style='color: red'>(+1 [+0.5%])</span> 186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/pairing-60d1b6a98c727974e7a1906c1e4b38970b018f7f.md) |<span style='color: red'>(+10 [+1.7%])</span> 608 |  1,745,757 | <span style='color: red'>(+2 [+1.0%])</span> 193 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/kitchen_sink-60d1b6a98c727974e7a1906c1e4b38970b018f7f.md) |<span style='color: green'>(-34 [-1.5%])</span> 2,183 |  2,579,903 | <span style='color: green'>(-7 [-1.5%])</span> 472 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/fibonacci_e2e-60d1b6a98c727974e7a1906c1e4b38970b018f7f.md) | 1,654 |  12,000,265 |  355 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/regex_e2e-60d1b6a98c727974e7a1906c1e4b38970b018f7f.md) | 769 |  4,137,067 |  206 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/ecrecover_e2e-60d1b6a98c727974e7a1906c1e4b38970b018f7f.md) | 496 |  123,583 |  174 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/pairing_e2e-60d1b6a98c727974e7a1906c1e4b38970b018f7f.md) | 648 |  1,745,757 |  182 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/kitchen_sink_e2e-60d1b6a98c727974e7a1906c1e4b38970b018f7f.md) | 2,702 |  2,579,903 |  467 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/60d1b6a98c727974e7a1906c1e4b38970b018f7f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29588368134)
