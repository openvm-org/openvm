| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-93afded000b4f03fccc2b654c7d83a6dfc62bb60.md) | 3,800 |  12,000,265 |  921 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-93afded000b4f03fccc2b654c7d83a6dfc62bb60.md) | 18,667 |  18,655,329 |  3,285 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-93afded000b4f03fccc2b654c7d83a6dfc62bb60.md) | 10,439 |  14,793,960 |  1,496 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-93afded000b4f03fccc2b654c7d83a6dfc62bb60.md) | 1,381 |  4,137,067 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-93afded000b4f03fccc2b654c7d83a6dfc62bb60.md) | 602 |  123,583 |  252 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-93afded000b4f03fccc2b654c7d83a6dfc62bb60.md) | 891 |  1,745,757 |  268 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-93afded000b4f03fccc2b654c7d83a6dfc62bb60.md) | 1,893 |  2,579,903 |  411 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-93afded000b4f03fccc2b654c7d83a6dfc62bb60.md) | 1,779 |  12,000,265 |  405 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-93afded000b4f03fccc2b654c7d83a6dfc62bb60.md) | 820 |  4,137,067 |  169 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-93afded000b4f03fccc2b654c7d83a6dfc62bb60.md) | 505 |  123,583 |  130 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-93afded000b4f03fccc2b654c7d83a6dfc62bb60.md) | 633 |  1,745,757 |  131 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-93afded000b4f03fccc2b654c7d83a6dfc62bb60.md) | 2,018 |  2,579,903 |  399 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/93afded000b4f03fccc2b654c7d83a6dfc62bb60

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25937400162)
