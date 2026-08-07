| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-1ca2f8b0b055a736077aa8c5ae007b9354dd9976.md) | 436 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-1ca2f8b0b055a736077aa8c5ae007b9354dd9976.md) | 7,177 |  14,365,133 |  1,609 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-1ca2f8b0b055a736077aa8c5ae007b9354dd9976.md) | 4,117 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-1ca2f8b0b055a736077aa8c5ae007b9354dd9976.md) | 732 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-1ca2f8b0b055a736077aa8c5ae007b9354dd9976.md) | 208 |  112,210 |  190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-1ca2f8b0b055a736077aa8c5ae007b9354dd9976.md) | 241 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-1ca2f8b0b055a736077aa8c5ae007b9354dd9976.md) | 2,161 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1ca2f8b0b055a736077aa8c5ae007b9354dd9976

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31220686713)
