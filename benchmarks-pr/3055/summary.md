| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/fibonacci-a551c3ba05de13e8335827c6a6b9227767ececd0.md) | 472 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/keccak-a551c3ba05de13e8335827c6a6b9227767ececd0.md) | 7,316 |  14,365,133 |  1,529 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/sha2_bench-a551c3ba05de13e8335827c6a6b9227767ececd0.md) | 4,785 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/regex-a551c3ba05de13e8335827c6a6b9227767ececd0.md) | 674 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/ecrecover-a551c3ba05de13e8335827c6a6b9227767ececd0.md) | 228 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/pairing-a551c3ba05de13e8335827c6a6b9227767ececd0.md) | 323 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/kitchen_sink-a551c3ba05de13e8335827c6a6b9227767ececd0.md) | 2,673 |  1,979,971 |  475 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a551c3ba05de13e8335827c6a6b9227767ececd0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30036299662)
