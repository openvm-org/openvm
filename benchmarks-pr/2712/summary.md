| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/fibonacci-75db1208725dec3e5d7c22b664fa32cd43596a45.md) | 3,854 |  12,000,265 |  956 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/keccak-75db1208725dec3e5d7c22b664fa32cd43596a45.md) | 18,579 |  18,655,329 |  3,302 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/sha2_bench-75db1208725dec3e5d7c22b664fa32cd43596a45.md) | 9,113 |  14,793,960 |  1,418 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/regex-75db1208725dec3e5d7c22b664fa32cd43596a45.md) | 1,426 |  4,137,067 |  380 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/ecrecover-75db1208725dec3e5d7c22b664fa32cd43596a45.md) | 644 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/pairing-75db1208725dec3e5d7c22b664fa32cd43596a45.md) | 904 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/kitchen_sink-75db1208725dec3e5d7c22b664fa32cd43596a45.md) | 2,095 |  2,579,903 |  439 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/75db1208725dec3e5d7c22b664fa32cd43596a45

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24796869937)
