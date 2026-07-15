| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/fibonacci-3436d5842c41e5a832e1f704e94b347af568b18e.md) | 407 |  4,000,051 |  226 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/keccak-3436d5842c41e5a832e1f704e94b347af568b18e.md) | 8,520 |  14,365,133 |  1,531 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/sha2_bench-3436d5842c41e5a832e1f704e94b347af568b18e.md) | 4,156 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/regex-3436d5842c41e5a832e1f704e94b347af568b18e.md) | 507 |  4,090,656 |  192 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/ecrecover-3436d5842c41e5a832e1f704e94b347af568b18e.md) | 222 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/pairing-3436d5842c41e5a832e1f704e94b347af568b18e.md) | 268 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/kitchen_sink-3436d5842c41e5a832e1f704e94b347af568b18e.md) | 1,879 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3436d5842c41e5a832e1f704e94b347af568b18e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29446827002)
