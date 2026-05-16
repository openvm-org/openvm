| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-dd9371cf4e670e337b3bcdf26e1ec15c64cdd8eb.md) | 1,413 |  4,000,051 |  437 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-dd9371cf4e670e337b3bcdf26e1ec15c64cdd8eb.md) | 13,431 |  14,365,133 |  2,216 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-dd9371cf4e670e337b3bcdf26e1ec15c64cdd8eb.md) | 8,984 |  11,167,961 |  1,410 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-dd9371cf4e670e337b3bcdf26e1ec15c64cdd8eb.md) | 1,342 |  4,090,656 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-dd9371cf4e670e337b3bcdf26e1ec15c64cdd8eb.md) | 466 |  112,210 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-dd9371cf4e670e337b3bcdf26e1ec15c64cdd8eb.md) | 601 |  592,827 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-dd9371cf4e670e337b3bcdf26e1ec15c64cdd8eb.md) | 2,202 |  1,979,971 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/dd9371cf4e670e337b3bcdf26e1ec15c64cdd8eb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25967215269)
