| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3128/fibonacci-67386a9015de11a4e3dcecf4870e007a172434f1.md) | 1,679 |  12,000,265 |  371 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3128/keccak-67386a9015de11a4e3dcecf4870e007a172434f1.md) | 9,659 |  18,655,329 |  1,544 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3128/sha2_bench-67386a9015de11a4e3dcecf4870e007a172434f1.md) | 5,236 |  14,793,960 |  584 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3128/regex-67386a9015de11a4e3dcecf4870e007a172434f1.md) | 697 |  4,137,067 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3128/ecrecover-67386a9015de11a4e3dcecf4870e007a172434f1.md) | 444 |  123,583 |  191 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3128/pairing-67386a9015de11a4e3dcecf4870e007a172434f1.md) | 586 |  1,745,757 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3128/kitchen_sink-67386a9015de11a4e3dcecf4870e007a172434f1.md) | 2,318 |  2,579,903 |  498 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/67386a9015de11a4e3dcecf4870e007a172434f1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32521463666)
