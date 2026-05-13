| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-d9d95dca3f003d1d82ee2c25e90818cdb3b6f659.md) | 1,884 |  4,000,051 |  539 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-d9d95dca3f003d1d82ee2c25e90818cdb3b6f659.md) | 13,727 |  14,365,133 |  2,260 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-d9d95dca3f003d1d82ee2c25e90818cdb3b6f659.md) | 9,449 |  11,167,961 |  1,406 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-d9d95dca3f003d1d82ee2c25e90818cdb3b6f659.md) | 1,609 |  4,090,656 |  380 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-d9d95dca3f003d1d82ee2c25e90818cdb3b6f659.md) | 644 |  112,210 |  289 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-d9d95dca3f003d1d82ee2c25e90818cdb3b6f659.md) | 753 |  592,827 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-d9d95dca3f003d1d82ee2c25e90818cdb3b6f659.md) | 2,053 |  1,979,971 |  439 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d9d95dca3f003d1d82ee2c25e90818cdb3b6f659

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25824479007)
