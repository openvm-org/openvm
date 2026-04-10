| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-71a0da5b4d0f9bba03d7c76475c6cd318626e510.md) | 3,881 |  12,000,265 |  971 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-71a0da5b4d0f9bba03d7c76475c6cd318626e510.md) | 18,533 |  18,655,329 |  3,321 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-71a0da5b4d0f9bba03d7c76475c6cd318626e510.md) | 9,928 |  14,793,960 |  1,406 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-71a0da5b4d0f9bba03d7c76475c6cd318626e510.md) | 1,413 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-71a0da5b4d0f9bba03d7c76475c6cd318626e510.md) | 641 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-71a0da5b4d0f9bba03d7c76475c6cd318626e510.md) | 912 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-71a0da5b4d0f9bba03d7c76475c6cd318626e510.md) | 2,150 |  2,579,903 |  437 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-71a0da5b4d0f9bba03d7c76475c6cd318626e510.md) | 1,719 |  12,000,265 |  455 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-71a0da5b4d0f9bba03d7c76475c6cd318626e510.md) | 854 |  4,137,067 |  194 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-71a0da5b4d0f9bba03d7c76475c6cd318626e510.md) | 407 |  123,583 |  152 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-71a0da5b4d0f9bba03d7c76475c6cd318626e510.md) | 515 |  1,745,757 |  151 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-71a0da5b4d0f9bba03d7c76475c6cd318626e510.md) | 2,182 |  2,579,903 |  423 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/71a0da5b4d0f9bba03d7c76475c6cd318626e510

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24265853430)
