| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/fibonacci-fd79a5eabda1de96df61b7f9b913f58ddbfc68c6.md) | 3,800 |  12,000,265 |  930 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/keccak-fd79a5eabda1de96df61b7f9b913f58ddbfc68c6.md) | 18,553 |  18,655,329 |  3,270 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/sha2_bench-fd79a5eabda1de96df61b7f9b913f58ddbfc68c6.md) | 10,104 |  14,793,960 |  1,466 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/regex-fd79a5eabda1de96df61b7f9b913f58ddbfc68c6.md) | 1,381 |  4,137,067 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/ecrecover-fd79a5eabda1de96df61b7f9b913f58ddbfc68c6.md) | 601 |  123,583 |  250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/pairing-fd79a5eabda1de96df61b7f9b913f58ddbfc68c6.md) | 884 |  1,745,757 |  264 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/kitchen_sink-fd79a5eabda1de96df61b7f9b913f58ddbfc68c6.md) | 1,897 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fd79a5eabda1de96df61b7f9b913f58ddbfc68c6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26235636974)
