| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/fibonacci-64d852f15523b087f38d24f291597c9fd2e15e14.md) | 471 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/keccak-64d852f15523b087f38d24f291597c9fd2e15e14.md) | 8,839 |  14,365,133 |  1,535 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/sha2_bench-64d852f15523b087f38d24f291597c9fd2e15e14.md) | 3,919 |  11,167,961 |  518 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/regex-64d852f15523b087f38d24f291597c9fd2e15e14.md) | 502 |  4,090,656 |  188 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/ecrecover-64d852f15523b087f38d24f291597c9fd2e15e14.md) | 219 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/pairing-64d852f15523b087f38d24f291597c9fd2e15e14.md) | 263 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/kitchen_sink-64d852f15523b087f38d24f291597c9fd2e15e14.md) | 1,904 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/64d852f15523b087f38d24f291597c9fd2e15e14

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29432767580)
