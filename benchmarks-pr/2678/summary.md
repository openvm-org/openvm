| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/fibonacci-276925102fa1cdd08e4c7f38eddfb41b6aa85c6f.md) | 3,781 |  12,000,265 |  948 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/keccak-276925102fa1cdd08e4c7f38eddfb41b6aa85c6f.md) | 18,747 |  18,655,329 |  3,358 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/regex-276925102fa1cdd08e4c7f38eddfb41b6aa85c6f.md) | 1,410 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/ecrecover-276925102fa1cdd08e4c7f38eddfb41b6aa85c6f.md) | 644 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/pairing-276925102fa1cdd08e4c7f38eddfb41b6aa85c6f.md) | 904 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/kitchen_sink-276925102fa1cdd08e4c7f38eddfb41b6aa85c6f.md) | 2,161 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/276925102fa1cdd08e4c7f38eddfb41b6aa85c6f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24165198729)
