| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/fibonacci-f71d9d610df763fbb0d09218209402ee9177a5b7.md) | 489 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/keccak-f71d9d610df763fbb0d09218209402ee9177a5b7.md) | 7,565 |  14,365,133 |  1,610 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/sha2_bench-f71d9d610df763fbb0d09218209402ee9177a5b7.md) | 4,402 |  11,167,961 |  531 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/regex-f71d9d610df763fbb0d09218209402ee9177a5b7.md) | 760 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/ecrecover-f71d9d610df763fbb0d09218209402ee9177a5b7.md) | 209 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/pairing-f71d9d610df763fbb0d09218209402ee9177a5b7.md) | 247 |  592,827 |  172 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/kitchen_sink-f71d9d610df763fbb0d09218209402ee9177a5b7.md) | 2,260 |  1,979,971 |  481 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f71d9d610df763fbb0d09218209402ee9177a5b7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/34330331472)
