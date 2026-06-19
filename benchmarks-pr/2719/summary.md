| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci-512f949f05f145b4f19a060d1477b59d0c6a0e9b.md) | 1,374 |  4,000,051 |  388 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/keccak-512f949f05f145b4f19a060d1477b59d0c6a0e9b.md) | 16,316 |  14,365,133 |  3,041 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/sha2_bench-512f949f05f145b4f19a060d1477b59d0c6a0e9b.md) | 10,106 |  11,167,961 |  1,012 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex-512f949f05f145b4f19a060d1477b59d0c6a0e9b.md) | 1,575 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover-512f949f05f145b4f19a060d1477b59d0c6a0e9b.md) | 442 |  112,210 |  313 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing-512f949f05f145b4f19a060d1477b59d0c6a0e9b.md) | 596 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink-512f949f05f145b4f19a060d1477b59d0c6a0e9b.md) | 3,895 |  1,979,971 |  858 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci_e2e-512f949f05f145b4f19a060d1477b59d0c6a0e9b.md) | 741 |  4,000,051 |  182 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex_e2e-512f949f05f145b4f19a060d1477b59d0c6a0e9b.md) | 973 |  4,090,656 |  168 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover_e2e-512f949f05f145b4f19a060d1477b59d0c6a0e9b.md) | 316 |  112,210 |  151 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing_e2e-512f949f05f145b4f19a060d1477b59d0c6a0e9b.md) | 410 |  592,827 |  142 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink_e2e-512f949f05f145b4f19a060d1477b59d0c6a0e9b.md) | 1,949 |  1,979,971 |  371 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/512f949f05f145b4f19a060d1477b59d0c6a0e9b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27815330354)
