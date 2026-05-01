| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/fibonacci-f96e810df249666c0b936eeafe73c9915362095e.md) | 4,306 |  12,000,265 |  1,327 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/keccak-f96e810df249666c0b936eeafe73c9915362095e.md) | 21,678 |  18,655,329 |  4,027 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/sha2_bench-f96e810df249666c0b936eeafe73c9915362095e.md) | 11,099 |  14,793,960 |  1,746 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/regex-f96e810df249666c0b936eeafe73c9915362095e.md) | 1,596 |  4,137,067 |  481 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/ecrecover-f96e810df249666c0b936eeafe73c9915362095e.md) | 670 |  123,583 |  354 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/pairing-f96e810df249666c0b936eeafe73c9915362095e.md) | 1,002 |  1,745,757 |  361 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/kitchen_sink-f96e810df249666c0b936eeafe73c9915362095e.md) | 2,343 |  2,579,903 |  659 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/fibonacci_e2e-f96e810df249666c0b936eeafe73c9915362095e.md) | 2,024 |  12,000,265 |  610 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/regex_e2e-f96e810df249666c0b936eeafe73c9915362095e.md) | 907 |  4,137,067 |  237 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/ecrecover_e2e-f96e810df249666c0b936eeafe73c9915362095e.md) | 557 |  123,583 |  188 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/pairing_e2e-f96e810df249666c0b936eeafe73c9915362095e.md) | 680 |  1,745,757 |  186 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/kitchen_sink_e2e-f96e810df249666c0b936eeafe73c9915362095e.md) | 2,435 |  2,579,903 |  642 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f96e810df249666c0b936eeafe73c9915362095e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25219180039)
