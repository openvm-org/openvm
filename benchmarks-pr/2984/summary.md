| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/fibonacci-44ad5818b8795cb1337e8071c94cd8bf2f559128.md) | 863 |  4,000,051 |  390 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/keccak-44ad5818b8795cb1337e8071c94cd8bf2f559128.md) | 15,318 |  14,365,133 |  3,012 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/sha2_bench-44ad5818b8795cb1337e8071c94cd8bf2f559128.md) | 8,093 |  11,167,961 |  1,008 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/regex-44ad5818b8795cb1337e8071c94cd8bf2f559128.md) | 1,030 |  4,090,656 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/ecrecover-44ad5818b8795cb1337e8071c94cd8bf2f559128.md) | 280 |  112,210 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/pairing-44ad5818b8795cb1337e8071c94cd8bf2f559128.md) | 366 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/kitchen_sink-44ad5818b8795cb1337e8071c94cd8bf2f559128.md) | 3,821 |  1,979,971 |  861 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/44ad5818b8795cb1337e8071c94cd8bf2f559128

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29033059092)
