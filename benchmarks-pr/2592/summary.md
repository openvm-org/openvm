| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-046a7a3df54d8496b3ffc58fa01456c281b1747d.md) | 3,871 |  12,000,265 |  959 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-046a7a3df54d8496b3ffc58fa01456c281b1747d.md) | 18,724 |  18,655,329 |  3,330 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-046a7a3df54d8496b3ffc58fa01456c281b1747d.md) | 8,975 |  14,793,960 |  1,393 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-046a7a3df54d8496b3ffc58fa01456c281b1747d.md) | 1,412 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-046a7a3df54d8496b3ffc58fa01456c281b1747d.md) | 641 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-046a7a3df54d8496b3ffc58fa01456c281b1747d.md) | 902 |  1,745,757 |  279 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-046a7a3df54d8496b3ffc58fa01456c281b1747d.md) | 2,092 |  2,579,903 |  433 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-046a7a3df54d8496b3ffc58fa01456c281b1747d.md) | 1,869 |  12,000,265 |  455 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-046a7a3df54d8496b3ffc58fa01456c281b1747d.md) | 850 |  4,137,067 |  192 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-046a7a3df54d8496b3ffc58fa01456c281b1747d.md) | 553 |  123,583 |  154 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-046a7a3df54d8496b3ffc58fa01456c281b1747d.md) | 656 |  1,745,757 |  153 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-046a7a3df54d8496b3ffc58fa01456c281b1747d.md) | 2,209 |  2,579,903 |  426 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/046a7a3df54d8496b3ffc58fa01456c281b1747d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24475894887)
