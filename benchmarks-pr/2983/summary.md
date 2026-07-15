| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/fibonacci-1036202d47de50f3416c3e130d5a9d88db33881a.md) | 403 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/keccak-1036202d47de50f3416c3e130d5a9d88db33881a.md) | 8,526 |  14,365,133 |  1,550 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/sha2_bench-1036202d47de50f3416c3e130d5a9d88db33881a.md) | 3,937 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/regex-1036202d47de50f3416c3e130d5a9d88db33881a.md) | 562 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/ecrecover-1036202d47de50f3416c3e130d5a9d88db33881a.md) | 219 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/pairing-1036202d47de50f3416c3e130d5a9d88db33881a.md) | 264 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/kitchen_sink-1036202d47de50f3416c3e130d5a9d88db33881a.md) | 1,911 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1036202d47de50f3416c3e130d5a9d88db33881a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29418080855)
