| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/fibonacci-ba56fabc67ce5cf858e9446c576752200a5e702b.md) | 461 |  4,000,051 |  239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/keccak-ba56fabc67ce5cf858e9446c576752200a5e702b.md) | 7,246 |  14,365,133 |  1,513 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/sha2_bench-ba56fabc67ce5cf858e9446c576752200a5e702b.md) | 4,700 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/regex-ba56fabc67ce5cf858e9446c576752200a5e702b.md) | 672 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/ecrecover-ba56fabc67ce5cf858e9446c576752200a5e702b.md) | 228 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/pairing-ba56fabc67ce5cf858e9446c576752200a5e702b.md) | 318 |  592,827 |  190 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/kitchen_sink-ba56fabc67ce5cf858e9446c576752200a5e702b.md) | 2,675 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ba56fabc67ce5cf858e9446c576752200a5e702b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29867218212)
