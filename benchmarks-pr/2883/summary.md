| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/fibonacci-3fe80367551b44993864e3539429940fa581ee3b.md) | 1,030 |  4,000,051 |  397 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/keccak-3fe80367551b44993864e3539429940fa581ee3b.md) | 16,468 |  14,365,133 |  3,061 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/sha2_bench-3fe80367551b44993864e3539429940fa581ee3b.md) | 8,122 |  11,167,961 |  994 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/regex-3fe80367551b44993864e3539429940fa581ee3b.md) | 1,215 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/ecrecover-3fe80367551b44993864e3539429940fa581ee3b.md) | 433 |  112,210 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/pairing-3fe80367551b44993864e3539429940fa581ee3b.md) | 594 |  592,827 |  298 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/kitchen_sink-3fe80367551b44993864e3539429940fa581ee3b.md) | 3,921 |  1,979,971 |  868 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3fe80367551b44993864e3539429940fa581ee3b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27826887783)
