| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2763/fibonacci-3412e9ac62682461b691f3ec8a5d5885b4e480d4.md) | 3,870 |  12,000,265 |  962 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2763/keccak-3412e9ac62682461b691f3ec8a5d5885b4e480d4.md) | 18,409 |  18,655,329 |  3,279 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2763/sha2_bench-3412e9ac62682461b691f3ec8a5d5885b4e480d4.md) | 8,974 |  14,793,960 |  1,387 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2763/regex-3412e9ac62682461b691f3ec8a5d5885b4e480d4.md) | 1,410 |  4,137,067 |  372 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2763/ecrecover-3412e9ac62682461b691f3ec8a5d5885b4e480d4.md) | 644 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2763/pairing-3412e9ac62682461b691f3ec8a5d5885b4e480d4.md) | 916 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2763/kitchen_sink-3412e9ac62682461b691f3ec8a5d5885b4e480d4.md) | 2,077 |  2,579,903 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3412e9ac62682461b691f3ec8a5d5885b4e480d4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25081652783)
