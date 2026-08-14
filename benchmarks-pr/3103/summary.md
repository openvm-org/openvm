| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-07dc0d28ec4dd4ab45bf700c8765e216880a3c2a.md) | 463 |  4,000,051 |  226 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-07dc0d28ec4dd4ab45bf700c8765e216880a3c2a.md) | 7,307 |  14,365,133 |  1,512 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-07dc0d28ec4dd4ab45bf700c8765e216880a3c2a.md) | 4,177 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-07dc0d28ec4dd4ab45bf700c8765e216880a3c2a.md) | 672 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-07dc0d28ec4dd4ab45bf700c8765e216880a3c2a.md) | 196 |  112,210 |  198 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-07dc0d28ec4dd4ab45bf700c8765e216880a3c2a.md) | 237 |  592,827 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-07dc0d28ec4dd4ab45bf700c8765e216880a3c2a.md) | 2,041 |  1,979,971 |  531 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/07dc0d28ec4dd4ab45bf700c8765e216880a3c2a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31814753620)
