| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/fibonacci-5d6c6daed8369b93c1621dadd5e0947b03e49a29.md) | 468 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/keccak-5d6c6daed8369b93c1621dadd5e0947b03e49a29.md) | 7,336 |  14,365,133 |  1,555 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/sha2_bench-5d6c6daed8369b93c1621dadd5e0947b03e49a29.md) | 4,692 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/regex-5d6c6daed8369b93c1621dadd5e0947b03e49a29.md) | 674 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/ecrecover-5d6c6daed8369b93c1621dadd5e0947b03e49a29.md) | 233 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/pairing-5d6c6daed8369b93c1621dadd5e0947b03e49a29.md) | 315 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/kitchen_sink-5d6c6daed8369b93c1621dadd5e0947b03e49a29.md) | 2,655 |  1,979,971 |  464 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5d6c6daed8369b93c1621dadd5e0947b03e49a29

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30129659642)
