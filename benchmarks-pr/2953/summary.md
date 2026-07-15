| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/fibonacci-762e22c6d4f2de11a80584012e4f5e5e8c387b69.md) | 442 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/keccak-762e22c6d4f2de11a80584012e4f5e5e8c387b69.md) | 8,368 |  14,365,133 |  1,511 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/sha2_bench-762e22c6d4f2de11a80584012e4f5e5e8c387b69.md) | 3,964 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/regex-762e22c6d4f2de11a80584012e4f5e5e8c387b69.md) | 580 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/ecrecover-762e22c6d4f2de11a80584012e4f5e5e8c387b69.md) | 220 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/pairing-762e22c6d4f2de11a80584012e4f5e5e8c387b69.md) | 263 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/kitchen_sink-762e22c6d4f2de11a80584012e4f5e5e8c387b69.md) | 1,884 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/762e22c6d4f2de11a80584012e4f5e5e8c387b69

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29406999341)
