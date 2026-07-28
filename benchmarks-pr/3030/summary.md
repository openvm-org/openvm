| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/fibonacci-0e23749f82954f71e3a20993794da7c78c0a9b60.md) |<span style='color: green'>(-6 [-0.4%])</span> 1,568 |  12,000,265 | <span style='color: green'>(-5 [-1.4%])</span> 358 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/keccak-0e23749f82954f71e3a20993794da7c78c0a9b60.md) | 9,267 |  18,655,329 | <span style='color: red'>(+8 [+0.5%])</span> 1,522 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/sha2_bench-0e23749f82954f71e3a20993794da7c78c0a9b60.md) | 4,945 |  14,793,960 | <span style='color: red'>(+4 [+0.7%])</span> 580 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/regex-0e23749f82954f71e3a20993794da7c78c0a9b60.md) |<span style='color: red'>(+1 [+0.2%])</span> 662 |  4,137,067 | <span style='color: red'>(+1 [+0.5%])</span> 213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/ecrecover-0e23749f82954f71e3a20993794da7c78c0a9b60.md) |<span style='color: red'>(+5 [+1.2%])</span> 436 |  123,583 | <span style='color: red'>(+4 [+2.2%])</span> 186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/pairing-0e23749f82954f71e3a20993794da7c78c0a9b60.md) |<span style='color: green'>(-16 [-2.8%])</span> 553 |  1,745,757 | <span style='color: green'>(-3 [-1.6%])</span> 187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/kitchen_sink-0e23749f82954f71e3a20993794da7c78c0a9b60.md) |<span style='color: red'>(+18 [+0.8%])</span> 2,221 |  2,579,903 | <span style='color: red'>(+3 [+0.6%])</span> 478 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0e23749f82954f71e3a20993794da7c78c0a9b60

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30373413927)
