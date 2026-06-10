| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-750c383cbb28063aa31b4c589af88ffc25f0ae2f.md) | 3,948 |  12,000,265 | <span style='color: green'>(-3338 [-74.4%])</span> 1,148 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-750c383cbb28063aa31b4c589af88ffc25f0ae2f.md) | 22,086 |  18,655,329 |  4,682 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-750c383cbb28063aa31b4c589af88ffc25f0ae2f.md) | 9,676 |  14,793,960 |  1,853 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-750c383cbb28063aa31b4c589af88ffc25f0ae2f.md) | 1,518 |  4,137,067 | <span style='color: green'>(-11562 [-96.4%])</span> 435 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-750c383cbb28063aa31b4c589af88ffc25f0ae2f.md) | 602 |  123,583 | <span style='color: green'>(-5575 [-95.2%])</span> 281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-750c383cbb28063aa31b4c589af88ffc25f0ae2f.md) | 940 |  1,745,757 | <span style='color: green'>(-6072 [-95.2%])</span> 308 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-750c383cbb28063aa31b4c589af88ffc25f0ae2f.md) | 4,157 |  2,579,903 |  887 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-750c383cbb28063aa31b4c589af88ffc25f0ae2f.md) | 1,710 |  12,000,265 |  496 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-750c383cbb28063aa31b4c589af88ffc25f0ae2f.md) | 720 |  4,137,067 |  195 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-750c383cbb28063aa31b4c589af88ffc25f0ae2f.md) | 364 |  123,583 |  142 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-750c383cbb28063aa31b4c589af88ffc25f0ae2f.md) | 504 |  1,745,757 |  149 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-750c383cbb28063aa31b4c589af88ffc25f0ae2f.md) | 2,158 |  2,579,903 |  383 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/750c383cbb28063aa31b4c589af88ffc25f0ae2f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27303071904)
