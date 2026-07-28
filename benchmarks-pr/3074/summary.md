| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/fibonacci-b522588661bb7de4835ef361814ff4c2d8dd1a43.md) | 497 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/keccak-b522588661bb7de4835ef361814ff4c2d8dd1a43.md) | 7,470 |  14,365,133 |  1,533 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/sha2_bench-b522588661bb7de4835ef361814ff4c2d8dd1a43.md) | 4,190 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/regex-b522588661bb7de4835ef361814ff4c2d8dd1a43.md) | 691 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/ecrecover-b522588661bb7de4835ef361814ff4c2d8dd1a43.md) | 232 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/pairing-b522588661bb7de4835ef361814ff4c2d8dd1a43.md) | 244 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/kitchen_sink-b522588661bb7de4835ef361814ff4c2d8dd1a43.md) | 2,044 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b522588661bb7de4835ef361814ff4c2d8dd1a43

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30346284711)
