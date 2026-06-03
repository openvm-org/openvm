| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/fibonacci-61e4be7b43223a8a3d3bffdd248f74316acc0b15.md) | 3,782 |  12,000,265 |  922 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/keccak-61e4be7b43223a8a3d3bffdd248f74316acc0b15.md) | 18,483 |  18,655,329 |  3,264 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/sha2_bench-61e4be7b43223a8a3d3bffdd248f74316acc0b15.md) | 10,113 |  14,793,960 |  1,449 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/regex-61e4be7b43223a8a3d3bffdd248f74316acc0b15.md) | 1,382 |  4,137,067 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/ecrecover-61e4be7b43223a8a3d3bffdd248f74316acc0b15.md) | 611 |  123,583 |  248 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/pairing-61e4be7b43223a8a3d3bffdd248f74316acc0b15.md) | 895 |  1,745,757 |  265 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/kitchen_sink-61e4be7b43223a8a3d3bffdd248f74316acc0b15.md) | 1,904 |  2,579,903 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/61e4be7b43223a8a3d3bffdd248f74316acc0b15

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26854946665)
