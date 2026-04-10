| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2699/fibonacci-30a13e26e57b2d54178fa3c7dc124b45c0be6782.md) | 3,839 |  12,000,265 | <span style='color: green'>(-9764 [-91.0%])</span> 960 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2699/keccak-30a13e26e57b2d54178fa3c7dc124b45c0be6782.md) | 18,541 |  18,655,329 |  3,316 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2699/sha2_bench-30a13e26e57b2d54178fa3c7dc124b45c0be6782.md) | 9,860 |  14,793,960 |  1,398 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2699/regex-30a13e26e57b2d54178fa3c7dc124b45c0be6782.md) | 1,417 |  4,137,067 | <span style='color: green'>(-27323 [-98.6%])</span> 382 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2699/ecrecover-30a13e26e57b2d54178fa3c7dc124b45c0be6782.md) | 645 |  123,583 | <span style='color: green'>(-10587 [-97.5%])</span> 270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2699/pairing-30a13e26e57b2d54178fa3c7dc124b45c0be6782.md) | 927 |  1,745,757 | <span style='color: green'>(-13856 [-97.9%])</span> 293 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2699/kitchen_sink-30a13e26e57b2d54178fa3c7dc124b45c0be6782.md) | 2,145 |  2,579,903 |  434 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2699/fibonacci_e2e-30a13e26e57b2d54178fa3c7dc124b45c0be6782.md) | 1,718 |  12,000,265 |  451 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2699/regex_e2e-30a13e26e57b2d54178fa3c7dc124b45c0be6782.md) | 853 |  4,137,067 |  192 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2699/ecrecover_e2e-30a13e26e57b2d54178fa3c7dc124b45c0be6782.md) | 407 |  123,583 |  152 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2699/pairing_e2e-30a13e26e57b2d54178fa3c7dc124b45c0be6782.md) | 515 |  1,745,757 |  152 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2699/kitchen_sink_e2e-30a13e26e57b2d54178fa3c7dc124b45c0be6782.md) | 2,203 |  2,579,903 |  430 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/30a13e26e57b2d54178fa3c7dc124b45c0be6782

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24267753523)
