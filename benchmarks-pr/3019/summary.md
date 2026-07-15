| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/fibonacci-80cd0964a554e04b8fe036c6129eb4a225a58b72.md) | 471 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/keccak-80cd0964a554e04b8fe036c6129eb4a225a58b72.md) | 8,765 |  14,365,133 |  1,538 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/sha2_bench-80cd0964a554e04b8fe036c6129eb4a225a58b72.md) | 3,937 |  11,167,961 |  519 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/regex-80cd0964a554e04b8fe036c6129eb4a225a58b72.md) | 500 |  4,090,656 |  193 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/ecrecover-80cd0964a554e04b8fe036c6129eb4a225a58b72.md) | 217 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/pairing-80cd0964a554e04b8fe036c6129eb4a225a58b72.md) | 259 |  592,827 |  181 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/kitchen_sink-80cd0964a554e04b8fe036c6129eb4a225a58b72.md) | 1,932 |  1,979,971 |  467 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/fibonacci_e2e-80cd0964a554e04b8fe036c6129eb4a225a58b72.md) | 477 |  4,000,051 |  218 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/regex_e2e-80cd0964a554e04b8fe036c6129eb4a225a58b72.md) | 583 |  4,090,656 |  181 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/ecrecover_e2e-80cd0964a554e04b8fe036c6129eb4a225a58b72.md) | 218 |  112,210 |  175 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/pairing_e2e-80cd0964a554e04b8fe036c6129eb4a225a58b72.md) | 275 |  592,827 |  174 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/kitchen_sink_e2e-80cd0964a554e04b8fe036c6129eb4a225a58b72.md) | 2,312 |  1,979,971 |  464 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/80cd0964a554e04b8fe036c6129eb4a225a58b72

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29445135472)
