| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-e74f9b1ee868c42776faf821b8b9f2e8829a7f68.md) | 1,695 |  12,000,265 |  370 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-e74f9b1ee868c42776faf821b8b9f2e8829a7f68.md) | 9,747 |  18,655,329 |  1,544 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-e74f9b1ee868c42776faf821b8b9f2e8829a7f68.md) | 5,244 |  14,793,960 |  585 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-e74f9b1ee868c42776faf821b8b9f2e8829a7f68.md) | 689 |  4,137,067 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-e74f9b1ee868c42776faf821b8b9f2e8829a7f68.md) | 434 |  123,583 |  190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-e74f9b1ee868c42776faf821b8b9f2e8829a7f68.md) | 624 |  1,745,757 |  195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-e74f9b1ee868c42776faf821b8b9f2e8829a7f68.md) | 2,306 |  2,579,903 |  499 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e74f9b1ee868c42776faf821b8b9f2e8829a7f68

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/38980213154)
