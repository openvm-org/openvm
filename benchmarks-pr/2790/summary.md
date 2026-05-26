| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci-a5d723b5da88953d0acd9d1c5ed68e832e03a4f2.md) | 3,785 |  12,000,265 |  924 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/keccak-a5d723b5da88953d0acd9d1c5ed68e832e03a4f2.md) | 18,884 |  18,655,329 |  3,309 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/sha2_bench-a5d723b5da88953d0acd9d1c5ed68e832e03a4f2.md) | 10,173 |  14,793,960 |  1,464 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex-a5d723b5da88953d0acd9d1c5ed68e832e03a4f2.md) | 1,385 |  4,137,067 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover-a5d723b5da88953d0acd9d1c5ed68e832e03a4f2.md) | 604 |  123,583 |  257 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing-a5d723b5da88953d0acd9d1c5ed68e832e03a4f2.md) | 901 |  1,745,757 |  266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink-a5d723b5da88953d0acd9d1c5ed68e832e03a4f2.md) | 1,919 |  2,579,903 |  414 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci_e2e-a5d723b5da88953d0acd9d1c5ed68e832e03a4f2.md) | 1,779 |  12,000,265 |  410 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex_e2e-a5d723b5da88953d0acd9d1c5ed68e832e03a4f2.md) | 810 |  4,137,067 |  169 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover_e2e-a5d723b5da88953d0acd9d1c5ed68e832e03a4f2.md) | 510 |  123,583 |  129 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing_e2e-a5d723b5da88953d0acd9d1c5ed68e832e03a4f2.md) | 633 |  1,745,757 |  132 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink_e2e-a5d723b5da88953d0acd9d1c5ed68e832e03a4f2.md) | 2,032 |  2,579,903 |  401 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a5d723b5da88953d0acd9d1c5ed68e832e03a4f2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26473999251)
