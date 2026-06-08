| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/fibonacci-f870ed85f8dee0be95662c746945a0af57f7a585.md) | 3,720 |  12,000,265 |  916 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/keccak-f870ed85f8dee0be95662c746945a0af57f7a585.md) | 18,046 |  18,655,329 |  3,274 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/sha2_bench-f870ed85f8dee0be95662c746945a0af57f7a585.md) | 10,005 |  14,793,960 |  1,463 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/regex-f870ed85f8dee0be95662c746945a0af57f7a585.md) | 1,410 |  4,137,067 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/ecrecover-f870ed85f8dee0be95662c746945a0af57f7a585.md) | 598 |  123,583 |  250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/pairing-f870ed85f8dee0be95662c746945a0af57f7a585.md) | 881 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/kitchen_sink-f870ed85f8dee0be95662c746945a0af57f7a585.md) | 3,823 |  2,579,903 |  944 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f870ed85f8dee0be95662c746945a0af57f7a585

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27169904447)
