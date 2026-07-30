| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/fibonacci-ac9b44ba551d039fd4aff1be2360fad1fb8c1f33.md) |<span style='color: green'>(-2 [-0.1%])</span> 1,572 |  12,000,265 | <span style='color: green'>(-3 [-0.8%])</span> 360 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/keccak-ac9b44ba551d039fd4aff1be2360fad1fb8c1f33.md) |<span style='color: green'>(-94 [-1.0%])</span> 9,180 |  18,655,329 | <span style='color: red'>(+14 [+0.9%])</span> 1,528 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/sha2_bench-ac9b44ba551d039fd4aff1be2360fad1fb8c1f33.md) |<span style='color: green'>(-35 [-0.7%])</span> 4,912 |  14,793,960 | <span style='color: red'>(+2 [+0.3%])</span> 578 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/regex-ac9b44ba551d039fd4aff1be2360fad1fb8c1f33.md) | 661 |  4,137,067 | <span style='color: green'>(-1 [-0.5%])</span> 211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/ecrecover-ac9b44ba551d039fd4aff1be2360fad1fb8c1f33.md) | 431 |  123,583 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/pairing-ac9b44ba551d039fd4aff1be2360fad1fb8c1f33.md) |<span style='color: red'>(+8 [+1.4%])</span> 577 |  1,745,757 | <span style='color: green'>(-2 [-1.1%])</span> 188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/kitchen_sink-ac9b44ba551d039fd4aff1be2360fad1fb8c1f33.md) | 2,201 |  2,579,903 | <span style='color: red'>(+4 [+0.8%])</span> 479 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ac9b44ba551d039fd4aff1be2360fad1fb8c1f33

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30556415378)
