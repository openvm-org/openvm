| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-6b3e714e0a4a3091f20ba3a47d8fb53c9b60093d.md) | 3,725 |  12,000,265 | <span style='color: green'>(-3569 [-79.6%])</span> 917 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-6b3e714e0a4a3091f20ba3a47d8fb53c9b60093d.md) | 17,972 |  18,655,329 |  3,261 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-6b3e714e0a4a3091f20ba3a47d8fb53c9b60093d.md) | 10,105 |  14,793,960 |  1,470 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-6b3e714e0a4a3091f20ba3a47d8fb53c9b60093d.md) | 1,405 |  4,137,067 | <span style='color: green'>(-11640 [-97.0%])</span> 357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-6b3e714e0a4a3091f20ba3a47d8fb53c9b60093d.md) | 601 |  123,583 | <span style='color: green'>(-5611 [-95.8%])</span> 245 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-6b3e714e0a4a3091f20ba3a47d8fb53c9b60093d.md) | 875 |  1,745,757 | <span style='color: green'>(-6119 [-95.9%])</span> 261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-6b3e714e0a4a3091f20ba3a47d8fb53c9b60093d.md) | 3,804 |  2,579,903 |  936 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-6b3e714e0a4a3091f20ba3a47d8fb53c9b60093d.md) | 1,618 |  12,000,265 |  412 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-6b3e714e0a4a3091f20ba3a47d8fb53c9b60093d.md) | 678 |  4,137,067 |  171 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-6b3e714e0a4a3091f20ba3a47d8fb53c9b60093d.md) | 361 |  123,583 |  131 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-6b3e714e0a4a3091f20ba3a47d8fb53c9b60093d.md) | 483 |  1,745,757 |  130 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-6b3e714e0a4a3091f20ba3a47d8fb53c9b60093d.md) | 1,825 |  2,579,903 |  399 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6b3e714e0a4a3091f20ba3a47d8fb53c9b60093d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27169916694)
