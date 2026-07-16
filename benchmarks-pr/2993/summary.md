| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-9c9dc3886787d71836b8416a318be9f04be8fde4.md) | 403 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-9c9dc3886787d71836b8416a318be9f04be8fde4.md) | 8,601 |  14,365,133 |  1,515 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-9c9dc3886787d71836b8416a318be9f04be8fde4.md) | 4,247 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-9c9dc3886787d71836b8416a318be9f04be8fde4.md) | 574 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-9c9dc3886787d71836b8416a318be9f04be8fde4.md) | 218 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-9c9dc3886787d71836b8416a318be9f04be8fde4.md) | 286 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-9c9dc3886787d71836b8416a318be9f04be8fde4.md) | 1,937 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9c9dc3886787d71836b8416a318be9f04be8fde4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29514733476)
