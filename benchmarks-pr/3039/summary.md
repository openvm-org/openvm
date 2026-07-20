| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/fibonacci-56f7d4fce5c7b43135bf2ea137828c48e7746b3e.md) | 411 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/keccak-56f7d4fce5c7b43135bf2ea137828c48e7746b3e.md) | 8,765 |  14,365,133 |  1,546 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/sha2_bench-56f7d4fce5c7b43135bf2ea137828c48e7746b3e.md) | 4,234 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/regex-56f7d4fce5c7b43135bf2ea137828c48e7746b3e.md) | 583 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/ecrecover-56f7d4fce5c7b43135bf2ea137828c48e7746b3e.md) | 224 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/pairing-56f7d4fce5c7b43135bf2ea137828c48e7746b3e.md) | 287 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/kitchen_sink-56f7d4fce5c7b43135bf2ea137828c48e7746b3e.md) | 1,945 |  1,979,971 |  466 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/56f7d4fce5c7b43135bf2ea137828c48e7746b3e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29760936365)
