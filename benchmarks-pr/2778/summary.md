| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-bc2f2d05609390bbd1d278b600e9a6ed6336f41a.md) | 1,625 |  4,000,051 |  540 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-bc2f2d05609390bbd1d278b600e9a6ed6336f41a.md) | 14,219 |  14,365,133 |  2,268 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-bc2f2d05609390bbd1d278b600e9a6ed6336f41a.md) | 9,381 |  11,167,961 |  1,408 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-bc2f2d05609390bbd1d278b600e9a6ed6336f41a.md) | 1,531 |  4,090,656 |  379 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-bc2f2d05609390bbd1d278b600e9a6ed6336f41a.md) | 517 |  112,210 |  296 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-bc2f2d05609390bbd1d278b600e9a6ed6336f41a.md) | 620 |  592,827 |  276 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-bc2f2d05609390bbd1d278b600e9a6ed6336f41a.md) | 1,969 |  1,979,971 |  430 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bc2f2d05609390bbd1d278b600e9a6ed6336f41a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25829346224)
