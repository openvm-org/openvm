| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/fibonacci-c3e83a8b4f77a28f4ddc09abd502c3d6b31a9e83.md) | 3,781 |  12,000,265 |  954 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/keccak-c3e83a8b4f77a28f4ddc09abd502c3d6b31a9e83.md) | 18,967 |  18,655,329 |  3,396 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/regex-c3e83a8b4f77a28f4ddc09abd502c3d6b31a9e83.md) | 1,420 |  4,137,067 |  381 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/ecrecover-c3e83a8b4f77a28f4ddc09abd502c3d6b31a9e83.md) | 645 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/pairing-c3e83a8b4f77a28f4ddc09abd502c3d6b31a9e83.md) | 904 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/kitchen_sink-c3e83a8b4f77a28f4ddc09abd502c3d6b31a9e83.md) | 2,155 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c3e83a8b4f77a28f4ddc09abd502c3d6b31a9e83

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24202688892)
