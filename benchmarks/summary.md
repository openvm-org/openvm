| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-3476851cc49135c76ee5eaa4d612f779d31260a0.md) | 1,664 |  12,000,265 |  369 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-3476851cc49135c76ee5eaa4d612f779d31260a0.md) | 9,645 |  18,655,329 |  1,551 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-3476851cc49135c76ee5eaa4d612f779d31260a0.md) | 5,255 |  14,793,960 |  591 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-3476851cc49135c76ee5eaa4d612f779d31260a0.md) | 693 |  4,137,067 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-3476851cc49135c76ee5eaa4d612f779d31260a0.md) | 439 |  123,583 |  193 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-3476851cc49135c76ee5eaa4d612f779d31260a0.md) | 587 |  1,745,757 |  197 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-3476851cc49135c76ee5eaa4d612f779d31260a0.md) | 2,295 |  2,579,903 |  499 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3476851cc49135c76ee5eaa4d612f779d31260a0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33108073127)
