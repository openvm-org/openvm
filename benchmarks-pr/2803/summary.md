| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/fibonacci-bc241671ac245c8a03cdaf6133c6fddad8fd3081.md) | 3,751 |  12,000,265 |  925 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/keccak-bc241671ac245c8a03cdaf6133c6fddad8fd3081.md) | 18,643 |  18,655,329 |  3,290 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/sha2_bench-bc241671ac245c8a03cdaf6133c6fddad8fd3081.md) | 10,235 |  14,793,960 |  1,465 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/regex-bc241671ac245c8a03cdaf6133c6fddad8fd3081.md) | 1,420 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/ecrecover-bc241671ac245c8a03cdaf6133c6fddad8fd3081.md) | 603 |  123,583 |  248 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/pairing-bc241671ac245c8a03cdaf6133c6fddad8fd3081.md) | 888 |  1,745,757 |  265 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/kitchen_sink-bc241671ac245c8a03cdaf6133c6fddad8fd3081.md) | 1,903 |  2,579,903 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bc241671ac245c8a03cdaf6133c6fddad8fd3081

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26470104348)
