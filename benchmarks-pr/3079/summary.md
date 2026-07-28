| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3079/fibonacci-cf434cd6d81c83a2d4d770de298d0bce94156511.md) | 469 |  4,000,051 |  239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3079/keccak-cf434cd6d81c83a2d4d770de298d0bce94156511.md) | 7,306 |  14,365,133 |  1,537 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3079/sha2_bench-cf434cd6d81c83a2d4d770de298d0bce94156511.md) | 4,761 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3079/regex-cf434cd6d81c83a2d4d770de298d0bce94156511.md) | 658 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3079/ecrecover-cf434cd6d81c83a2d4d770de298d0bce94156511.md) | 226 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3079/pairing-cf434cd6d81c83a2d4d770de298d0bce94156511.md) | 318 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3079/kitchen_sink-cf434cd6d81c83a2d4d770de298d0bce94156511.md) | 2,611 |  1,979,971 |  467 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/cf434cd6d81c83a2d4d770de298d0bce94156511

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30366624503)
