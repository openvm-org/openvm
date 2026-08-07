| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/fibonacci-c73d0f8c884a5c10f68850b3d4d68762348e3593.md) | 474 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/keccak-c73d0f8c884a5c10f68850b3d4d68762348e3593.md) | 7,310 |  14,365,133 |  1,515 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/sha2_bench-c73d0f8c884a5c10f68850b3d4d68762348e3593.md) | 4,168 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/regex-c73d0f8c884a5c10f68850b3d4d68762348e3593.md) | 662 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/ecrecover-c73d0f8c884a5c10f68850b3d4d68762348e3593.md) | 250 |  78,475 |  227 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/pairing-c73d0f8c884a5c10f68850b3d4d68762348e3593.md) | 230 |  592,827 |  198 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/kitchen_sink-c73d0f8c884a5c10f68850b3d4d68762348e3593.md) | 2,337 |  2,341,811 |  556 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c73d0f8c884a5c10f68850b3d4d68762348e3593

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31192170654)
