| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3049/fibonacci-2b1dee3592bf2b98f165f10d1029416db317e623.md) | 404 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3049/keccak-2b1dee3592bf2b98f165f10d1029416db317e623.md) | 8,587 |  14,365,133 |  1,522 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3049/sha2_bench-2b1dee3592bf2b98f165f10d1029416db317e623.md) | 4,236 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3049/regex-2b1dee3592bf2b98f165f10d1029416db317e623.md) | 568 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3049/ecrecover-2b1dee3592bf2b98f165f10d1029416db317e623.md) | 218 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3049/pairing-2b1dee3592bf2b98f165f10d1029416db317e623.md) | 290 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3049/kitchen_sink-2b1dee3592bf2b98f165f10d1029416db317e623.md) | 1,904 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2b1dee3592bf2b98f165f10d1029416db317e623

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29696134140)
