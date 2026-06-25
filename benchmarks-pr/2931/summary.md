| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/fibonacci-1a2a97f4a2976a6caf2d2459430b75293ef15ccd.md) | 1,038 |  4,000,051 |  398 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/keccak-1a2a97f4a2976a6caf2d2459430b75293ef15ccd.md) | 16,222 |  14,365,133 |  3,025 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/sha2_bench-1a2a97f4a2976a6caf2d2459430b75293ef15ccd.md) | 8,066 |  11,167,961 |  989 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/regex-1a2a97f4a2976a6caf2d2459430b75293ef15ccd.md) | 1,171 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/ecrecover-1a2a97f4a2976a6caf2d2459430b75293ef15ccd.md) | 440 |  112,210 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/pairing-1a2a97f4a2976a6caf2d2459430b75293ef15ccd.md) | 584 |  592,827 |  293 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/kitchen_sink-1a2a97f4a2976a6caf2d2459430b75293ef15ccd.md) | 3,851 |  1,979,971 |  853 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1a2a97f4a2976a6caf2d2459430b75293ef15ccd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28173962866)
