| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3046/fibonacci-b32d0d6967799a759db195caf1ccdf2f706b3452.md) | 415 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3046/keccak-b32d0d6967799a759db195caf1ccdf2f706b3452.md) | 8,656 |  14,365,133 |  1,540 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3046/sha2_bench-b32d0d6967799a759db195caf1ccdf2f706b3452.md) | 4,246 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3046/regex-b32d0d6967799a759db195caf1ccdf2f706b3452.md) | 576 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3046/ecrecover-b32d0d6967799a759db195caf1ccdf2f706b3452.md) | 219 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3046/pairing-b32d0d6967799a759db195caf1ccdf2f706b3452.md) | 293 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3046/kitchen_sink-b32d0d6967799a759db195caf1ccdf2f706b3452.md) | 1,921 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b32d0d6967799a759db195caf1ccdf2f706b3452

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29684576692)
