| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/fibonacci-8e5d0c7310d9003d6084767c7a46b3de6fcebbd5.md) | 3,842 |  12,000,265 |  950 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/keccak-8e5d0c7310d9003d6084767c7a46b3de6fcebbd5.md) | 18,405 |  18,655,329 |  3,283 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/sha2_bench-8e5d0c7310d9003d6084767c7a46b3de6fcebbd5.md) | 8,983 |  14,793,960 |  1,401 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/regex-8e5d0c7310d9003d6084767c7a46b3de6fcebbd5.md) | 1,426 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/ecrecover-8e5d0c7310d9003d6084767c7a46b3de6fcebbd5.md) | 650 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/pairing-8e5d0c7310d9003d6084767c7a46b3de6fcebbd5.md) | 912 |  1,745,757 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/kitchen_sink-8e5d0c7310d9003d6084767c7a46b3de6fcebbd5.md) | 2,097 |  2,579,903 |  435 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8e5d0c7310d9003d6084767c7a46b3de6fcebbd5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24595088471)
