| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-a5750efb86696daeab68fe152ab624e430f69e40.md) | 3,827 |  12,000,265 |  952 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-a5750efb86696daeab68fe152ab624e430f69e40.md) | 18,439 |  18,655,329 |  3,320 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-a5750efb86696daeab68fe152ab624e430f69e40.md) | 1,413 |  4,137,067 |  380 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-a5750efb86696daeab68fe152ab624e430f69e40.md) | 645 |  123,583 |  265 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-a5750efb86696daeab68fe152ab624e430f69e40.md) | 914 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-a5750efb86696daeab68fe152ab624e430f69e40.md) | 2,162 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a5750efb86696daeab68fe152ab624e430f69e40

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24156295247)
