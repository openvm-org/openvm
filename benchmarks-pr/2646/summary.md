| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/fibonacci-d9dd0f48a029e426afbc87219b844f0c0bc25632.md) | 3,848 |  12,000,265 |  958 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/keccak-d9dd0f48a029e426afbc87219b844f0c0bc25632.md) | 18,566 |  18,655,329 |  3,327 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/regex-d9dd0f48a029e426afbc87219b844f0c0bc25632.md) | 1,421 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/ecrecover-d9dd0f48a029e426afbc87219b844f0c0bc25632.md) | 735 |  317,792 |  357 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/pairing-d9dd0f48a029e426afbc87219b844f0c0bc25632.md) | 912 |  1,745,757 |  311 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/kitchen_sink-d9dd0f48a029e426afbc87219b844f0c0bc25632.md) | 2,391 |  2,580,026 |  788 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d9dd0f48a029e426afbc87219b844f0c0bc25632

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24184718836)
