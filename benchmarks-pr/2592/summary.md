| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-b8194ef7a93d5abead2ef60d2ce9a38e2b287521.md) | 3,750 |  12,000,265 |  913 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-b8194ef7a93d5abead2ef60d2ce9a38e2b287521.md) | 18,516 |  18,655,329 |  3,278 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-b8194ef7a93d5abead2ef60d2ce9a38e2b287521.md) | 10,204 |  14,793,960 |  1,472 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-b8194ef7a93d5abead2ef60d2ce9a38e2b287521.md) | 1,396 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-b8194ef7a93d5abead2ef60d2ce9a38e2b287521.md) | 601 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-b8194ef7a93d5abead2ef60d2ce9a38e2b287521.md) | 896 |  1,745,757 |  263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-b8194ef7a93d5abead2ef60d2ce9a38e2b287521.md) | 1,884 |  2,579,903 |  409 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-b8194ef7a93d5abead2ef60d2ce9a38e2b287521.md) | 1,634 |  12,000,265 |  413 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-b8194ef7a93d5abead2ef60d2ce9a38e2b287521.md) | 676 |  4,137,067 |  169 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-b8194ef7a93d5abead2ef60d2ce9a38e2b287521.md) | 360 |  123,583 |  129 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-b8194ef7a93d5abead2ef60d2ce9a38e2b287521.md) | 489 |  1,745,757 |  133 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-b8194ef7a93d5abead2ef60d2ce9a38e2b287521.md) | 1,867 |  2,579,903 |  399 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b8194ef7a93d5abead2ef60d2ce9a38e2b287521

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26294211664)
