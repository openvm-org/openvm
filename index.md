| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-8a34b6637e5aee888fcc5a6c13f1109d52da22e7.md) | 1,682 |  12,000,265 |  370 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-8a34b6637e5aee888fcc5a6c13f1109d52da22e7.md) | 9,735 |  18,655,329 |  1,555 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-8a34b6637e5aee888fcc5a6c13f1109d52da22e7.md) | 5,247 |  14,793,960 |  585 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-8a34b6637e5aee888fcc5a6c13f1109d52da22e7.md) | 689 |  4,137,067 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-8a34b6637e5aee888fcc5a6c13f1109d52da22e7.md) | 429 |  123,583 |  191 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-8a34b6637e5aee888fcc5a6c13f1109d52da22e7.md) | 581 |  1,745,757 |  194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-8a34b6637e5aee888fcc5a6c13f1109d52da22e7.md) | 2,314 |  2,579,903 |  499 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8a34b6637e5aee888fcc5a6c13f1109d52da22e7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/40719775252)
