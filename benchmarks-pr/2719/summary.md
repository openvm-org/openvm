| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci-53784108cf93718b7acd062c0b5a910a82f40156.md) | 1,659 |  4,000,051 |  532 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/keccak-53784108cf93718b7acd062c0b5a910a82f40156.md) | 16,284 |  14,365,133 |  3,029 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/sha2_bench-53784108cf93718b7acd062c0b5a910a82f40156.md) | 10,437 |  11,167,961 |  1,934 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex-53784108cf93718b7acd062c0b5a910a82f40156.md) | 1,536 |  4,090,656 |  428 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover-53784108cf93718b7acd062c0b5a910a82f40156.md) | 484 |  112,210 |  310 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing-53784108cf93718b7acd062c0b5a910a82f40156.md) | 623 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink-53784108cf93718b7acd062c0b5a910a82f40156.md) | 3,939 |  1,979,971 |  864 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci_e2e-53784108cf93718b7acd062c0b5a910a82f40156.md) | 845 |  4,000,051 |  233 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex_e2e-53784108cf93718b7acd062c0b5a910a82f40156.md) | 839 |  4,090,656 |  200 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover_e2e-53784108cf93718b7acd062c0b5a910a82f40156.md) | 332 |  112,210 |  152 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing_e2e-53784108cf93718b7acd062c0b5a910a82f40156.md) | 407 |  592,827 |  144 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink_e2e-53784108cf93718b7acd062c0b5a910a82f40156.md) | 1,951 |  1,979,971 |  372 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/53784108cf93718b7acd062c0b5a910a82f40156

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27493501577)
