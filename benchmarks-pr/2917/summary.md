| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/fibonacci-f65d8fc2d0375eb82a31fafd2eb7ffcd58c2ec4e.md) | 1,038 |  4,000,051 |  388 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/keccak-f65d8fc2d0375eb82a31fafd2eb7ffcd58c2ec4e.md) | 16,234 |  14,365,133 |  2,999 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/sha2_bench-f65d8fc2d0375eb82a31fafd2eb7ffcd58c2ec4e.md) | 8,164 |  11,167,961 |  994 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/regex-f65d8fc2d0375eb82a31fafd2eb7ffcd58c2ec4e.md) | 1,239 |  4,090,656 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/ecrecover-f65d8fc2d0375eb82a31fafd2eb7ffcd58c2ec4e.md) | 434 |  112,210 |  285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/pairing-f65d8fc2d0375eb82a31fafd2eb7ffcd58c2ec4e.md) | 602 |  592,827 |  295 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/kitchen_sink-f65d8fc2d0375eb82a31fafd2eb7ffcd58c2ec4e.md) | 3,838 |  1,979,971 |  848 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f65d8fc2d0375eb82a31fafd2eb7ffcd58c2ec4e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27935468901)
