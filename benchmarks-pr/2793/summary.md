| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/fibonacci-03d7be5e5b57f8d7159446d0b3a498ff99a8bea9.md) | 3,797 |  12,000,265 |  928 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/keccak-03d7be5e5b57f8d7159446d0b3a498ff99a8bea9.md) | 18,827 |  18,655,329 |  3,331 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/sha2_bench-03d7be5e5b57f8d7159446d0b3a498ff99a8bea9.md) | 10,135 |  14,793,960 |  1,446 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/regex-03d7be5e5b57f8d7159446d0b3a498ff99a8bea9.md) | 1,383 |  4,137,067 |  349 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/ecrecover-03d7be5e5b57f8d7159446d0b3a498ff99a8bea9.md) | 601 |  123,583 |  252 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/pairing-03d7be5e5b57f8d7159446d0b3a498ff99a8bea9.md) | 895 |  1,745,757 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/kitchen_sink-03d7be5e5b57f8d7159446d0b3a498ff99a8bea9.md) | 1,890 |  2,579,903 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/03d7be5e5b57f8d7159446d0b3a498ff99a8bea9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26130536617)
