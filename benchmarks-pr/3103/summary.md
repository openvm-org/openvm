| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-b1520706759c65e8f90a41ddc2d92a623661cb89.md) | 462 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-b1520706759c65e8f90a41ddc2d92a623661cb89.md) | 7,331 |  14,365,133 |  1,515 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-b1520706759c65e8f90a41ddc2d92a623661cb89.md) | 4,173 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-b1520706759c65e8f90a41ddc2d92a623661cb89.md) | 667 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-b1520706759c65e8f90a41ddc2d92a623661cb89.md) | 196 |  112,210 |  196 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-b1520706759c65e8f90a41ddc2d92a623661cb89.md) | 236 |  592,827 |  195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-b1520706759c65e8f90a41ddc2d92a623661cb89.md) | 2,024 |  1,979,971 |  526 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b1520706759c65e8f90a41ddc2d92a623661cb89

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31729389004)
