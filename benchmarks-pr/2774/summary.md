| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/fibonacci-25622fe58350edc84891dd5941f1e32d489b1f9f.md) | 3,815 |  12,000,265 |  948 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/keccak-25622fe58350edc84891dd5941f1e32d489b1f9f.md) | 18,733 |  18,655,329 |  3,355 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/sha2_bench-25622fe58350edc84891dd5941f1e32d489b1f9f.md) | 8,970 |  14,793,960 |  1,404 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/regex-25622fe58350edc84891dd5941f1e32d489b1f9f.md) | 1,404 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/ecrecover-25622fe58350edc84891dd5941f1e32d489b1f9f.md) | 634 |  123,583 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/pairing-25622fe58350edc84891dd5941f1e32d489b1f9f.md) | 894 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/kitchen_sink-25622fe58350edc84891dd5941f1e32d489b1f9f.md) | 2,088 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/25622fe58350edc84891dd5941f1e32d489b1f9f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25673536658)
