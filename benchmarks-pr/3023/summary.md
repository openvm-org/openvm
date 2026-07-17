| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/fibonacci-b244edf19fd2ec9a24a3415c555665f98b687c98.md) | 413 |  4,000,051 |  236 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/keccak-b244edf19fd2ec9a24a3415c555665f98b687c98.md) | 8,413 |  14,365,133 |  1,515 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/sha2_bench-b244edf19fd2ec9a24a3415c555665f98b687c98.md) | 4,073 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/regex-b244edf19fd2ec9a24a3415c555665f98b687c98.md) | 495 |  4,090,656 |  191 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/ecrecover-b244edf19fd2ec9a24a3415c555665f98b687c98.md) | 225 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/pairing-b244edf19fd2ec9a24a3415c555665f98b687c98.md) | 280 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/kitchen_sink-b244edf19fd2ec9a24a3415c555665f98b687c98.md) | 1,877 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b244edf19fd2ec9a24a3415c555665f98b687c98

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29541491236)
