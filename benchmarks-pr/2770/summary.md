| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/fibonacci-b29a991f6991517415450269c28e3498e4aec446.md) | 3,804 |  12,000,265 |  946 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/keccak-b29a991f6991517415450269c28e3498e4aec446.md) | 18,605 |  18,655,329 |  3,313 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/sha2_bench-b29a991f6991517415450269c28e3498e4aec446.md) | 9,055 |  14,793,960 |  1,413 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/regex-b29a991f6991517415450269c28e3498e4aec446.md) | 1,415 |  4,137,067 |  379 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/ecrecover-b29a991f6991517415450269c28e3498e4aec446.md) | 643 |  123,583 |  276 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/pairing-b29a991f6991517415450269c28e3498e4aec446.md) | 910 |  1,745,757 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/kitchen_sink-b29a991f6991517415450269c28e3498e4aec446.md) | 2,098 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b29a991f6991517415450269c28e3498e4aec446

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25238662926)
