| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/fibonacci-2ee8d4da49e090aaeee41b0e0c5fe6cdc1848b1e.md) | 3,862 |  12,000,265 |  950 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/keccak-2ee8d4da49e090aaeee41b0e0c5fe6cdc1848b1e.md) | 18,728 |  18,655,329 |  3,335 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/sha2_bench-2ee8d4da49e090aaeee41b0e0c5fe6cdc1848b1e.md) | 9,035 |  14,793,960 |  1,416 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/regex-2ee8d4da49e090aaeee41b0e0c5fe6cdc1848b1e.md) | 1,441 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/ecrecover-2ee8d4da49e090aaeee41b0e0c5fe6cdc1848b1e.md) | 650 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/pairing-2ee8d4da49e090aaeee41b0e0c5fe6cdc1848b1e.md) | 916 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/kitchen_sink-2ee8d4da49e090aaeee41b0e0c5fe6cdc1848b1e.md) | 2,087 |  2,579,903 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2ee8d4da49e090aaeee41b0e0c5fe6cdc1848b1e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24594493951)
