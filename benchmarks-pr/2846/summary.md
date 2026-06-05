| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-e823766aa7d6aac1a6fce90c7137cee53fcab9d9.md) | 3,734 |  12,000,265 | <span style='color: green'>(-3567 [-79.5%])</span> 919 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-e823766aa7d6aac1a6fce90c7137cee53fcab9d9.md) | 18,133 |  18,655,329 |  3,288 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-e823766aa7d6aac1a6fce90c7137cee53fcab9d9.md) | 9,924 |  14,793,960 |  1,456 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-e823766aa7d6aac1a6fce90c7137cee53fcab9d9.md) | 1,397 |  4,137,067 | <span style='color: green'>(-11639 [-97.0%])</span> 358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-e823766aa7d6aac1a6fce90c7137cee53fcab9d9.md) | 600 |  123,583 | <span style='color: green'>(-5609 [-95.8%])</span> 247 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-e823766aa7d6aac1a6fce90c7137cee53fcab9d9.md) | 897 |  1,745,757 | <span style='color: green'>(-6110 [-95.8%])</span> 270 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-e823766aa7d6aac1a6fce90c7137cee53fcab9d9.md) | 3,872 |  2,579,903 |  954 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-e823766aa7d6aac1a6fce90c7137cee53fcab9d9.md) | 1,626 |  12,000,265 |  409 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-e823766aa7d6aac1a6fce90c7137cee53fcab9d9.md) | 668 |  4,137,067 |  171 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-e823766aa7d6aac1a6fce90c7137cee53fcab9d9.md) | 364 |  123,583 |  130 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-e823766aa7d6aac1a6fce90c7137cee53fcab9d9.md) | 480 |  1,745,757 |  131 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-e823766aa7d6aac1a6fce90c7137cee53fcab9d9.md) | 1,828 |  2,579,903 |  403 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e823766aa7d6aac1a6fce90c7137cee53fcab9d9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27030575924)
