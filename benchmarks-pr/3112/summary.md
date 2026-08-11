| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/fibonacci-6c05f2b6eea2a320cbf42fb1197af58d7b3939b0.md) | 469 |  4,000,051 |  236 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/keccak-6c05f2b6eea2a320cbf42fb1197af58d7b3939b0.md) | 7,382 |  14,365,133 |  1,526 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/sha2_bench-6c05f2b6eea2a320cbf42fb1197af58d7b3939b0.md) | 4,206 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/regex-6c05f2b6eea2a320cbf42fb1197af58d7b3939b0.md) | 670 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/ecrecover-6c05f2b6eea2a320cbf42fb1197af58d7b3939b0.md) | 201 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/pairing-6c05f2b6eea2a320cbf42fb1197af58d7b3939b0.md) | 233 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/kitchen_sink-6c05f2b6eea2a320cbf42fb1197af58d7b3939b0.md) | 2,009 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6c05f2b6eea2a320cbf42fb1197af58d7b3939b0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31525255559)
