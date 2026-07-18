| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/fibonacci-83cf21b2192107aa982a84b6a15704abf186fa55.md) | 411 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/keccak-83cf21b2192107aa982a84b6a15704abf186fa55.md) | 8,704 |  14,365,133 |  1,533 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/sha2_bench-83cf21b2192107aa982a84b6a15704abf186fa55.md) | 4,217 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/regex-83cf21b2192107aa982a84b6a15704abf186fa55.md) | 583 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/ecrecover-83cf21b2192107aa982a84b6a15704abf186fa55.md) | 222 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/pairing-83cf21b2192107aa982a84b6a15704abf186fa55.md) | 293 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/kitchen_sink-83cf21b2192107aa982a84b6a15704abf186fa55.md) | 1,922 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/83cf21b2192107aa982a84b6a15704abf186fa55

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29658421727)
