| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-6b7e3c87fc3e02e077a47e5994e7585c6844c24c.md) | 476 |  4,000,051 |  243 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-6b7e3c87fc3e02e077a47e5994e7585c6844c24c.md) | 7,122 |  14,365,133 |  1,547 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-6b7e3c87fc3e02e077a47e5994e7585c6844c24c.md) | 4,491 |  11,167,961 |  531 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-6b7e3c87fc3e02e077a47e5994e7585c6844c24c.md) | 672 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-6b7e3c87fc3e02e077a47e5994e7585c6844c24c.md) | 221 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-6b7e3c87fc3e02e077a47e5994e7585c6844c24c.md) | 250 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-6b7e3c87fc3e02e077a47e5994e7585c6844c24c.md) | 2,715 |  1,979,971 |  472 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6b7e3c87fc3e02e077a47e5994e7585c6844c24c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29486805481)
