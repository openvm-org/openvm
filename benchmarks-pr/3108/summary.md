| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/fibonacci-5cfb843f8bfe0acdea7481ae6c8306319992ecf1.md) | 440 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/keccak-5cfb843f8bfe0acdea7481ae6c8306319992ecf1.md) | 7,152 |  14,365,133 |  1,589 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/sha2_bench-5cfb843f8bfe0acdea7481ae6c8306319992ecf1.md) | 4,126 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/regex-5cfb843f8bfe0acdea7481ae6c8306319992ecf1.md) | 712 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/ecrecover-5cfb843f8bfe0acdea7481ae6c8306319992ecf1.md) | 209 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/pairing-5cfb843f8bfe0acdea7481ae6c8306319992ecf1.md) | 233 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/kitchen_sink-5cfb843f8bfe0acdea7481ae6c8306319992ecf1.md) | 2,144 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5cfb843f8bfe0acdea7481ae6c8306319992ecf1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31205567224)
