| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/fibonacci-f39798600e8ff2a3d1caad6e0827fe12e6281d4f.md) | 1,586 |  12,000,265 |  361 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/keccak-f39798600e8ff2a3d1caad6e0827fe12e6281d4f.md) | 9,343 |  18,655,329 |  1,527 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/sha2_bench-f39798600e8ff2a3d1caad6e0827fe12e6281d4f.md) | 4,949 |  14,793,960 |  576 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/regex-f39798600e8ff2a3d1caad6e0827fe12e6281d4f.md) | 666 |  4,137,067 |  210 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/ecrecover-f39798600e8ff2a3d1caad6e0827fe12e6281d4f.md) | 435 |  123,583 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/pairing-f39798600e8ff2a3d1caad6e0827fe12e6281d4f.md) | 563 |  1,745,757 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/kitchen_sink-f39798600e8ff2a3d1caad6e0827fe12e6281d4f.md) | 2,218 |  2,579,903 |  480 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f39798600e8ff2a3d1caad6e0827fe12e6281d4f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31196373720)
