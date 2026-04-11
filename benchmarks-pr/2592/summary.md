| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-7b8e917c7936b91f5b1d1559e50fae4d3c9bbc51.md) | 3,847 |  12,000,265 |  965 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-7b8e917c7936b91f5b1d1559e50fae4d3c9bbc51.md) | 18,706 |  18,655,329 |  3,346 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-7b8e917c7936b91f5b1d1559e50fae4d3c9bbc51.md) | 9,881 |  14,793,960 |  1,402 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-7b8e917c7936b91f5b1d1559e50fae4d3c9bbc51.md) | 1,437 |  4,137,067 |  380 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-7b8e917c7936b91f5b1d1559e50fae4d3c9bbc51.md) | 649 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-7b8e917c7936b91f5b1d1559e50fae4d3c9bbc51.md) | 903 |  1,745,757 |  279 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-7b8e917c7936b91f5b1d1559e50fae4d3c9bbc51.md) | 2,174 |  2,579,903 |  438 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-7b8e917c7936b91f5b1d1559e50fae4d3c9bbc51.md) | 1,722 |  12,000,265 |  457 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-7b8e917c7936b91f5b1d1559e50fae4d3c9bbc51.md) | 861 |  4,137,067 |  194 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-7b8e917c7936b91f5b1d1559e50fae4d3c9bbc51.md) | 408 |  123,583 |  150 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-7b8e917c7936b91f5b1d1559e50fae4d3c9bbc51.md) | 514 |  1,745,757 |  152 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-7b8e917c7936b91f5b1d1559e50fae4d3c9bbc51.md) | 2,191 |  2,579,903 |  427 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7b8e917c7936b91f5b1d1559e50fae4d3c9bbc51

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24289830863)
