| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-3faceb64adac207a7a97d17f96d642d4ea13b9e5.md) | 1,881 |  4,000,051 |  534 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-3faceb64adac207a7a97d17f96d642d4ea13b9e5.md) | 13,474 |  14,365,133 |  2,215 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-3faceb64adac207a7a97d17f96d642d4ea13b9e5.md) | 9,575 |  11,167,961 |  1,430 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-3faceb64adac207a7a97d17f96d642d4ea13b9e5.md) | 1,597 |  4,090,656 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-3faceb64adac207a7a97d17f96d642d4ea13b9e5.md) | 638 |  112,210 |  291 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-3faceb64adac207a7a97d17f96d642d4ea13b9e5.md) | 758 |  592,827 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-3faceb64adac207a7a97d17f96d642d4ea13b9e5.md) | 2,025 |  1,979,971 |  425 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3faceb64adac207a7a97d17f96d642d4ea13b9e5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25867889809)
