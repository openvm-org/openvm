| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-e10f770902d10e9cf763c9fac28e286a5de8d929.md) | 477 |  4,000,051 |  240 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-e10f770902d10e9cf763c9fac28e286a5de8d929.md) | 7,307 |  14,365,133 |  1,530 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-e10f770902d10e9cf763c9fac28e286a5de8d929.md) | 4,727 |  11,167,961 |  531 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-e10f770902d10e9cf763c9fac28e286a5de8d929.md) | 686 |  4,090,656 |  224 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-e10f770902d10e9cf763c9fac28e286a5de8d929.md) | 229 |  112,210 |  192 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-e10f770902d10e9cf763c9fac28e286a5de8d929.md) | 277 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-e10f770902d10e9cf763c9fac28e286a5de8d929.md) | 2,740 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e10f770902d10e9cf763c9fac28e286a5de8d929

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29986085830)
