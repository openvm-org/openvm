| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci-3131d09ca3de19adecbb351e563ebfd92b60b4e0.md) | 1,553 |  4,000,051 |  434 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/keccak-3131d09ca3de19adecbb351e563ebfd92b60b4e0.md) | 13,667 |  14,365,133 |  2,398 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/sha2_bench-3131d09ca3de19adecbb351e563ebfd92b60b4e0.md) | 9,023 |  11,167,961 |  1,426 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex-3131d09ca3de19adecbb351e563ebfd92b60b4e0.md) | 1,544 |  4,090,656 |  348 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover-3131d09ca3de19adecbb351e563ebfd92b60b4e0.md) | 493 |  112,210 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing-3131d09ca3de19adecbb351e563ebfd92b60b4e0.md) | 600 |  592,827 |  259 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink-3131d09ca3de19adecbb351e563ebfd92b60b4e0.md) | 3,795 |  1,979,971 |  950 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci_e2e-3131d09ca3de19adecbb351e563ebfd92b60b4e0.md) | 812 |  4,000,051 |  198 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex_e2e-3131d09ca3de19adecbb351e563ebfd92b60b4e0.md) | 904 |  4,090,656 |  169 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover_e2e-3131d09ca3de19adecbb351e563ebfd92b60b4e0.md) | 329 |  112,210 |  135 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing_e2e-3131d09ca3de19adecbb351e563ebfd92b60b4e0.md) | 397 |  592,827 |  126 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink_e2e-3131d09ca3de19adecbb351e563ebfd92b60b4e0.md) | 2,060 |  1,979,971 |  390 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3131d09ca3de19adecbb351e563ebfd92b60b4e0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26963124954)
