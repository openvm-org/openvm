| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/fibonacci-2ec5bc457d232e4ae06399ef8bf2d6e98fbc4940.md) | 1,028 |  4,000,051 |  396 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/keccak-2ec5bc457d232e4ae06399ef8bf2d6e98fbc4940.md) | 16,041 |  14,365,133 |  2,996 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/sha2_bench-2ec5bc457d232e4ae06399ef8bf2d6e98fbc4940.md) | 8,044 |  11,167,961 |  992 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/regex-2ec5bc457d232e4ae06399ef8bf2d6e98fbc4940.md) | 1,173 |  4,090,656 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/ecrecover-2ec5bc457d232e4ae06399ef8bf2d6e98fbc4940.md) | 440 |  112,210 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/pairing-2ec5bc457d232e4ae06399ef8bf2d6e98fbc4940.md) | 582 |  592,827 |  290 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/kitchen_sink-2ec5bc457d232e4ae06399ef8bf2d6e98fbc4940.md) | 3,870 |  1,979,971 |  861 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2ec5bc457d232e4ae06399ef8bf2d6e98fbc4940

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28152365073)
