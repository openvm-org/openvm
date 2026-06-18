| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-b9d416af1fc0ce10b7cec747a5147c3a9edcec4b.md) | 3,241 |  12,000,265 | <span style='color: green'>(-3781 [-84.3%])</span> 705 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-b9d416af1fc0ce10b7cec747a5147c3a9edcec4b.md) | 16,450 |  18,655,329 |  3,048 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-b9d416af1fc0ce10b7cec747a5147c3a9edcec4b.md) | 9,079 |  14,793,960 |  1,102 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-b9d416af1fc0ce10b7cec747a5147c3a9edcec4b.md) | 1,163 |  4,137,067 | <span style='color: green'>(-11646 [-97.1%])</span> 351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-b9d416af1fc0ce10b7cec747a5147c3a9edcec4b.md) | 600 |  123,583 | <span style='color: green'>(-5571 [-95.1%])</span> 285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-b9d416af1fc0ce10b7cec747a5147c3a9edcec4b.md) | 934 |  1,745,757 | <span style='color: green'>(-6075 [-95.2%])</span> 305 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-b9d416af1fc0ce10b7cec747a5147c3a9edcec4b.md) | 4,103 |  2,579,903 |  878 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-b9d416af1fc0ce10b7cec747a5147c3a9edcec4b.md) | 1,390 |  12,000,265 |  289 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-b9d416af1fc0ce10b7cec747a5147c3a9edcec4b.md) | 637 |  4,137,067 |  167 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-b9d416af1fc0ce10b7cec747a5147c3a9edcec4b.md) | 393 |  123,583 |  141 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-b9d416af1fc0ce10b7cec747a5147c3a9edcec4b.md) | 528 |  1,745,757 |  148 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-b9d416af1fc0ce10b7cec747a5147c3a9edcec4b.md) | 1,988 |  2,579,903 |  385 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b9d416af1fc0ce10b7cec747a5147c3a9edcec4b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27790590869)
