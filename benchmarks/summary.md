| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-059d5c59ce00be393653cc5b63eb15314c69a43d.md) | 3,052 |  12,000,265 |  679 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-059d5c59ce00be393653cc5b63eb15314c69a43d.md) | 16,343 |  18,655,329 |  3,030 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-059d5c59ce00be393653cc5b63eb15314c69a43d.md) | 9,175 |  14,793,960 |  1,107 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-059d5c59ce00be393653cc5b63eb15314c69a43d.md) | 1,161 |  4,137,067 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-059d5c59ce00be393653cc5b63eb15314c69a43d.md) | 599 |  123,583 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-059d5c59ce00be393653cc5b63eb15314c69a43d.md) | 934 |  1,745,757 |  308 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-059d5c59ce00be393653cc5b63eb15314c69a43d.md) | 4,117 |  2,579,903 |  879 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/059d5c59ce00be393653cc5b63eb15314c69a43d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29415666932)
