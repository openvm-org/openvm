| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/fibonacci-3b562907edae931410762a6162096c27a31122f7.md) | 465 |  4,000,051 |  227 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/keccak-3b562907edae931410762a6162096c27a31122f7.md) | 8,833 |  14,365,133 |  1,537 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/sha2_bench-3b562907edae931410762a6162096c27a31122f7.md) | 3,954 |  11,167,961 |  519 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/regex-3b562907edae931410762a6162096c27a31122f7.md) | 506 |  4,090,656 |  189 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/ecrecover-3b562907edae931410762a6162096c27a31122f7.md) | 217 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/pairing-3b562907edae931410762a6162096c27a31122f7.md) | 266 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/kitchen_sink-3b562907edae931410762a6162096c27a31122f7.md) | 1,917 |  1,979,971 |  467 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/fibonacci_e2e-3b562907edae931410762a6162096c27a31122f7.md) | 477 |  4,000,051 |  218 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/regex_e2e-3b562907edae931410762a6162096c27a31122f7.md) | 581 |  4,090,656 |  184 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/ecrecover_e2e-3b562907edae931410762a6162096c27a31122f7.md) | 216 |  112,210 |  174 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/pairing_e2e-3b562907edae931410762a6162096c27a31122f7.md) | 277 |  592,827 |  173 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/kitchen_sink_e2e-3b562907edae931410762a6162096c27a31122f7.md) | 2,264 |  1,979,971 |  456 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3b562907edae931410762a6162096c27a31122f7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29400267318)
