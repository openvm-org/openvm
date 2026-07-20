| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/fibonacci-e76c5388c9684d7a282808be99e7a12769acdcf3.md) | 407 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/keccak-e76c5388c9684d7a282808be99e7a12769acdcf3.md) | 8,762 |  14,365,133 |  1,546 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/sha2_bench-e76c5388c9684d7a282808be99e7a12769acdcf3.md) | 4,223 |  11,167,961 |  518 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/regex-e76c5388c9684d7a282808be99e7a12769acdcf3.md) | 582 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/ecrecover-e76c5388c9684d7a282808be99e7a12769acdcf3.md) | 222 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/pairing-e76c5388c9684d7a282808be99e7a12769acdcf3.md) | 285 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/kitchen_sink-e76c5388c9684d7a282808be99e7a12769acdcf3.md) | 1,928 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e76c5388c9684d7a282808be99e7a12769acdcf3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29760960384)
