| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-8072f6996b80e9de85d3fb2d5f2c0c3a5ab9f4be.md) | 883 |  4,000,051 |  397 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-8072f6996b80e9de85d3fb2d5f2c0c3a5ab9f4be.md) | 15,598 |  14,365,133 |  3,023 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-8072f6996b80e9de85d3fb2d5f2c0c3a5ab9f4be.md) | 7,978 |  11,167,961 |  1,019 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-8072f6996b80e9de85d3fb2d5f2c0c3a5ab9f4be.md) | 1,032 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-8072f6996b80e9de85d3fb2d5f2c0c3a5ab9f4be.md) | 304 |  112,210 |  293 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-8072f6996b80e9de85d3fb2d5f2c0c3a5ab9f4be.md) | 433 |  592,827 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-8072f6996b80e9de85d3fb2d5f2c0c3a5ab9f4be.md) | 3,704 |  1,979,971 |  860 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8072f6996b80e9de85d3fb2d5f2c0c3a5ab9f4be

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29272977855)
