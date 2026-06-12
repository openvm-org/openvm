| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-3784813f5068dfbeee8c5ff8f3a7f78561ce2bd0.md) | 3,978 |  12,000,265 | <span style='color: green'>(-3342 [-74.5%])</span> 1,144 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-3784813f5068dfbeee8c5ff8f3a7f78561ce2bd0.md) | 21,786 |  18,655,329 |  4,623 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-3784813f5068dfbeee8c5ff8f3a7f78561ce2bd0.md) | 9,522 |  14,793,960 |  1,826 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-3784813f5068dfbeee8c5ff8f3a7f78561ce2bd0.md) | 1,489 |  4,137,067 | <span style='color: green'>(-11567 [-96.4%])</span> 430 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-3784813f5068dfbeee8c5ff8f3a7f78561ce2bd0.md) | 599 |  123,583 | <span style='color: green'>(-5575 [-95.2%])</span> 281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-3784813f5068dfbeee8c5ff8f3a7f78561ce2bd0.md) | 944 |  1,745,757 | <span style='color: green'>(-6072 [-95.2%])</span> 308 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-3784813f5068dfbeee8c5ff8f3a7f78561ce2bd0.md) | 4,104 |  2,579,903 |  874 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-3784813f5068dfbeee8c5ff8f3a7f78561ce2bd0.md) | 1,712 |  12,000,265 |  496 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-3784813f5068dfbeee8c5ff8f3a7f78561ce2bd0.md) | 713 |  4,137,067 |  197 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-3784813f5068dfbeee8c5ff8f3a7f78561ce2bd0.md) | 367 |  123,583 |  142 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-3784813f5068dfbeee8c5ff8f3a7f78561ce2bd0.md) | 504 |  1,745,757 |  147 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-3784813f5068dfbeee8c5ff8f3a7f78561ce2bd0.md) | 2,165 |  2,579,903 |  384 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3784813f5068dfbeee8c5ff8f3a7f78561ce2bd0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27367131407)
