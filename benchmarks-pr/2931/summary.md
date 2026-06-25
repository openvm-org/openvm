| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/fibonacci-a8ebcc76c4d625b948573aa0a5e13480fc031cb0.md) | 1,021 |  4,000,051 |  395 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/keccak-a8ebcc76c4d625b948573aa0a5e13480fc031cb0.md) | 16,177 |  14,365,133 |  3,031 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/sha2_bench-a8ebcc76c4d625b948573aa0a5e13480fc031cb0.md) | 8,048 |  11,167,961 |  992 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/regex-a8ebcc76c4d625b948573aa0a5e13480fc031cb0.md) | 1,185 |  4,090,656 |  365 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/ecrecover-a8ebcc76c4d625b948573aa0a5e13480fc031cb0.md) | 437 |  112,210 |  276 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/pairing-a8ebcc76c4d625b948573aa0a5e13480fc031cb0.md) | 584 |  592,827 |  300 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/kitchen_sink-a8ebcc76c4d625b948573aa0a5e13480fc031cb0.md) | 3,863 |  1,979,971 |  860 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a8ebcc76c4d625b948573aa0a5e13480fc031cb0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28142647021)
