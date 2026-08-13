| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-3f41c1f256ed8d40608ee1a46c266a121d7bf8db.md) | 462 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-3f41c1f256ed8d40608ee1a46c266a121d7bf8db.md) | 7,426 |  14,365,133 |  1,533 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-3f41c1f256ed8d40608ee1a46c266a121d7bf8db.md) | 4,178 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-3f41c1f256ed8d40608ee1a46c266a121d7bf8db.md) | 664 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-3f41c1f256ed8d40608ee1a46c266a121d7bf8db.md) | 194 |  112,210 |  196 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-3f41c1f256ed8d40608ee1a46c266a121d7bf8db.md) | 238 |  592,827 |  199 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-3f41c1f256ed8d40608ee1a46c266a121d7bf8db.md) | 2,033 |  1,979,971 |  530 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3f41c1f256ed8d40608ee1a46c266a121d7bf8db

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31671518011)
