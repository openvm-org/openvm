| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2814/fibonacci-2f6e3b2ab13329f90b01ebaee4c244bb4e1b7e7e.md) | 3,782 |  12,000,265 |  923 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2814/keccak-2f6e3b2ab13329f90b01ebaee4c244bb4e1b7e7e.md) | 18,573 |  18,655,329 |  3,290 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2814/sha2_bench-2f6e3b2ab13329f90b01ebaee4c244bb4e1b7e7e.md) | 10,245 |  14,793,960 |  1,463 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2814/regex-2f6e3b2ab13329f90b01ebaee4c244bb4e1b7e7e.md) | 1,414 |  4,137,067 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2814/ecrecover-2f6e3b2ab13329f90b01ebaee4c244bb4e1b7e7e.md) | 610 |  123,583 |  246 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2814/pairing-2f6e3b2ab13329f90b01ebaee4c244bb4e1b7e7e.md) | 886 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2814/kitchen_sink-2f6e3b2ab13329f90b01ebaee4c244bb4e1b7e7e.md) | 1,893 |  2,579,903 |  408 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2f6e3b2ab13329f90b01ebaee4c244bb4e1b7e7e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26458045698)
