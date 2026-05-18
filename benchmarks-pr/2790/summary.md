| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci-c8a20927c800d0f4af0a13b9f3fbb07e9776b42a.md) | 3,772 |  12,000,265 |  921 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/keccak-c8a20927c800d0f4af0a13b9f3fbb07e9776b42a.md) | 18,420 |  18,655,329 |  3,245 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/sha2_bench-c8a20927c800d0f4af0a13b9f3fbb07e9776b42a.md) | 10,085 |  14,793,960 |  1,445 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex-c8a20927c800d0f4af0a13b9f3fbb07e9776b42a.md) | 1,423 |  4,137,067 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover-c8a20927c800d0f4af0a13b9f3fbb07e9776b42a.md) | 598 |  123,583 |  252 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing-c8a20927c800d0f4af0a13b9f3fbb07e9776b42a.md) | 910 |  1,745,757 |  271 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink-c8a20927c800d0f4af0a13b9f3fbb07e9776b42a.md) | 1,889 |  2,579,903 |  410 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci_e2e-c8a20927c800d0f4af0a13b9f3fbb07e9776b42a.md) | 1,778 |  12,000,265 |  415 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex_e2e-c8a20927c800d0f4af0a13b9f3fbb07e9776b42a.md) | 824 |  4,137,067 |  170 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover_e2e-c8a20927c800d0f4af0a13b9f3fbb07e9776b42a.md) | 513 |  123,583 |  130 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing_e2e-c8a20927c800d0f4af0a13b9f3fbb07e9776b42a.md) | 631 |  1,745,757 |  130 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink_e2e-c8a20927c800d0f4af0a13b9f3fbb07e9776b42a.md) | 2,019 |  2,579,903 |  403 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c8a20927c800d0f4af0a13b9f3fbb07e9776b42a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26055811782)
