| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/fibonacci-cbe4307fd691404e794bd8c8ea7c01638c6512e8.md) | 464 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/keccak-cbe4307fd691404e794bd8c8ea7c01638c6512e8.md) | 7,214 |  14,365,133 |  1,580 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/sha2_bench-cbe4307fd691404e794bd8c8ea7c01638c6512e8.md) | 4,022 |  11,167,961 |  519 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/regex-cbe4307fd691404e794bd8c8ea7c01638c6512e8.md) | 716 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/ecrecover-cbe4307fd691404e794bd8c8ea7c01638c6512e8.md) | 205 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/pairing-cbe4307fd691404e794bd8c8ea7c01638c6512e8.md) | 240 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/kitchen_sink-cbe4307fd691404e794bd8c8ea7c01638c6512e8.md) | 2,142 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/cbe4307fd691404e794bd8c8ea7c01638c6512e8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32501161487)
