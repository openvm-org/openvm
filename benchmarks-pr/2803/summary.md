| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/fibonacci-fcc282b2fe276a8c9a01c504ff9822d624dc4f34.md) | 3,728 |  12,000,265 |  908 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/keccak-fcc282b2fe276a8c9a01c504ff9822d624dc4f34.md) | 18,407 |  18,655,329 |  3,255 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/sha2_bench-fcc282b2fe276a8c9a01c504ff9822d624dc4f34.md) | 10,172 |  14,793,960 |  1,447 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/regex-fcc282b2fe276a8c9a01c504ff9822d624dc4f34.md) | 1,398 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/ecrecover-fcc282b2fe276a8c9a01c504ff9822d624dc4f34.md) | 605 |  123,583 |  244 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/pairing-fcc282b2fe276a8c9a01c504ff9822d624dc4f34.md) | 882 |  1,745,757 |  259 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/kitchen_sink-fcc282b2fe276a8c9a01c504ff9822d624dc4f34.md) | 1,907 |  2,579,903 |  410 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fcc282b2fe276a8c9a01c504ff9822d624dc4f34

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26308342643)
