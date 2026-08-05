| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/fibonacci-f4bb0742f85839f734e720a63b8854cef20d9865.md) | 475 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/keccak-f4bb0742f85839f734e720a63b8854cef20d9865.md) | 7,291 |  14,365,133 |  1,520 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/sha2_bench-f4bb0742f85839f734e720a63b8854cef20d9865.md) | 4,147 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/regex-f4bb0742f85839f734e720a63b8854cef20d9865.md) | 659 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/ecrecover-f4bb0742f85839f734e720a63b8854cef20d9865.md) | 224 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/pairing-f4bb0742f85839f734e720a63b8854cef20d9865.md) | 236 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/kitchen_sink-f4bb0742f85839f734e720a63b8854cef20d9865.md) | 2,044 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f4bb0742f85839f734e720a63b8854cef20d9865

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30987842057)
