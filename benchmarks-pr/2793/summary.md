| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/fibonacci-135025cdc9b4abc301e2ce792bd5f2c3d43c1ed5.md) | 3,797 |  12,000,265 |  923 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/keccak-135025cdc9b4abc301e2ce792bd5f2c3d43c1ed5.md) | 18,803 |  18,655,329 |  3,322 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/sha2_bench-135025cdc9b4abc301e2ce792bd5f2c3d43c1ed5.md) | 10,179 |  14,793,960 |  1,452 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/regex-135025cdc9b4abc301e2ce792bd5f2c3d43c1ed5.md) | 1,424 |  4,137,067 |  363 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/ecrecover-135025cdc9b4abc301e2ce792bd5f2c3d43c1ed5.md) | 600 |  123,583 |  252 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/pairing-135025cdc9b4abc301e2ce792bd5f2c3d43c1ed5.md) | 888 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/kitchen_sink-135025cdc9b4abc301e2ce792bd5f2c3d43c1ed5.md) | 1,905 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/135025cdc9b4abc301e2ce792bd5f2c3d43c1ed5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26128923634)
