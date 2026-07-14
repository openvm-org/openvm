| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/fibonacci-6e1f16212b2d87e749c6bcf9be02f5f8a8f8bc94.md) | 477 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/keccak-6e1f16212b2d87e749c6bcf9be02f5f8a8f8bc94.md) | 8,639 |  14,365,133 |  1,544 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/sha2_bench-6e1f16212b2d87e749c6bcf9be02f5f8a8f8bc94.md) | 4,077 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/regex-6e1f16212b2d87e749c6bcf9be02f5f8a8f8bc94.md) | 559 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/ecrecover-6e1f16212b2d87e749c6bcf9be02f5f8a8f8bc94.md) | 218 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/pairing-6e1f16212b2d87e749c6bcf9be02f5f8a8f8bc94.md) | 294 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/kitchen_sink-6e1f16212b2d87e749c6bcf9be02f5f8a8f8bc94.md) | 1,958 |  1,979,971 |  466 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/fibonacci_e2e-6e1f16212b2d87e749c6bcf9be02f5f8a8f8bc94.md) | 498 |  4,000,051 |  220 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/regex_e2e-6e1f16212b2d87e749c6bcf9be02f5f8a8f8bc94.md) | 649 |  4,090,656 |  203 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/ecrecover_e2e-6e1f16212b2d87e749c6bcf9be02f5f8a8f8bc94.md) | 220 |  112,210 |  173 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/pairing_e2e-6e1f16212b2d87e749c6bcf9be02f5f8a8f8bc94.md) | 307 |  592,827 |  175 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/kitchen_sink_e2e-6e1f16212b2d87e749c6bcf9be02f5f8a8f8bc94.md) | 2,332 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6e1f16212b2d87e749c6bcf9be02f5f8a8f8bc94

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29331464583)
