| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/fibonacci-8e42409552438627a9b583e7cd6fb2efe7476fd2.md) | 472 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/keccak-8e42409552438627a9b583e7cd6fb2efe7476fd2.md) | 7,314 |  14,365,133 |  1,531 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/sha2_bench-8e42409552438627a9b583e7cd6fb2efe7476fd2.md) | 4,777 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/regex-8e42409552438627a9b583e7cd6fb2efe7476fd2.md) | 675 |  4,090,656 |  222 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/ecrecover-8e42409552438627a9b583e7cd6fb2efe7476fd2.md) | 231 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/pairing-8e42409552438627a9b583e7cd6fb2efe7476fd2.md) | 322 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/kitchen_sink-8e42409552438627a9b583e7cd6fb2efe7476fd2.md) | 2,663 |  1,979,971 |  473 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8e42409552438627a9b583e7cd6fb2efe7476fd2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30128068408)
