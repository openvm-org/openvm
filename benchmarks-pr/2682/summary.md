| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/fibonacci-f8b09b6b5d3158fef52fda64ff763197f1498a6e.md) | 3,814 |  12,000,265 |  950 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/keccak-f8b09b6b5d3158fef52fda64ff763197f1498a6e.md) | 18,890 |  18,655,329 |  3,362 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/regex-f8b09b6b5d3158fef52fda64ff763197f1498a6e.md) | 1,416 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/ecrecover-f8b09b6b5d3158fef52fda64ff763197f1498a6e.md) | 642 |  123,583 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/pairing-f8b09b6b5d3158fef52fda64ff763197f1498a6e.md) | 905 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/kitchen_sink-f8b09b6b5d3158fef52fda64ff763197f1498a6e.md) | 2,153 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f8b09b6b5d3158fef52fda64ff763197f1498a6e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24206237233)
