| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3119/fibonacci-8890ef90d5bf6fc655db77a4e2df34fba23ea4cc.md) | 462 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3119/keccak-8890ef90d5bf6fc655db77a4e2df34fba23ea4cc.md) | 7,489 |  14,365,133 |  1,545 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3119/sha2_bench-8890ef90d5bf6fc655db77a4e2df34fba23ea4cc.md) | 4,130 |  11,167,961 |  514 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3119/regex-8890ef90d5bf6fc655db77a4e2df34fba23ea4cc.md) | 672 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3119/ecrecover-8890ef90d5bf6fc655db77a4e2df34fba23ea4cc.md) | 199 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3119/pairing-8890ef90d5bf6fc655db77a4e2df34fba23ea4cc.md) | 232 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3119/kitchen_sink-8890ef90d5bf6fc655db77a4e2df34fba23ea4cc.md) | 2,024 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8890ef90d5bf6fc655db77a4e2df34fba23ea4cc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31687791850)
