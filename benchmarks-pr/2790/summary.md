| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci-4e8d623ca3c3defc70c13897081a06306c929272.md) | 3,807 |  12,000,265 |  921 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/keccak-4e8d623ca3c3defc70c13897081a06306c929272.md) | 18,980 |  18,655,329 |  3,349 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/sha2_bench-4e8d623ca3c3defc70c13897081a06306c929272.md) | 10,111 |  14,793,960 |  1,448 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex-4e8d623ca3c3defc70c13897081a06306c929272.md) | 1,401 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover-4e8d623ca3c3defc70c13897081a06306c929272.md) | 599 |  123,583 |  252 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing-4e8d623ca3c3defc70c13897081a06306c929272.md) | 892 |  1,745,757 |  269 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink-4e8d623ca3c3defc70c13897081a06306c929272.md) | 1,901 |  2,579,903 |  410 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci_e2e-4e8d623ca3c3defc70c13897081a06306c929272.md) | 1,778 |  12,000,265 |  412 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex_e2e-4e8d623ca3c3defc70c13897081a06306c929272.md) | 817 |  4,137,067 |  171 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover_e2e-4e8d623ca3c3defc70c13897081a06306c929272.md) | 512 |  123,583 |  130 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing_e2e-4e8d623ca3c3defc70c13897081a06306c929272.md) | 634 |  1,745,757 |  132 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink_e2e-4e8d623ca3c3defc70c13897081a06306c929272.md) | 2,033 |  2,579,903 |  403 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4e8d623ca3c3defc70c13897081a06306c929272

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26918765853)
